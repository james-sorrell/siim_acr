"""
Created on Mon Aug 10 15:09:37 2020
Data Helper Functions and Classes to process SIIM ACR Data

https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/data

The data is comprised of images in DICOM format and annotations in the form of image IDs 
and run-length-encoded (RLE) masks. Some of the images contain instances of pneumothorax 
(collapsed lung), which are indicated by encoded binary masks in the annotations. 
Some training images have multiple annotations.

@author: James Sorrell

Data Preparation Inspired By:
    https://www.kaggle.com/ekhtiar/lung-segmentation-cropping-resunet-tf-keras
    https://www.kaggle.com/ekhtiar/finding-pneumo-part-1-eda-and-unet

"""

import random
import pydicom
import config as c
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2

# Import Mask Functions from Provided Files
# TODO: It would be cleaner to do this as a 
# module import, but I need to get that working
# cleanly with Syder, for now this will do
import sys
sys.path.insert(0, '../input')
from mask_functions import rle2mask, mask2rle

class DataLoader():
    """ Data Controller Class
    This class will handle data parsing and preparation.
    """ 

    def __init__(self):
        c.debugPrint("Data Controller Created", 1)
        self.rle = pd.read_csv('../input/data/train-rle.csv')
        self.rle.columns = ['ImageId', 'EncodedPixels']
        c.debugPrint("\tLoading training data...", 1)
        self.train_data = self.package_data('../input/data/dicom-images-train/*/*/*.dcm', type="train")
        c.debugPrint("\tLoading testing data...", 1)
        self.test_data = self.package_data('../input/data/dicom-images-test/*/*/*.dcm', type="test")

    def package_data(self, path, type):
        """ Generate training data """
        fns = sorted(glob(path))
        metadata_df = pd.DataFrame()
        metadata_list = []
        for file_path in tqdm(fns):
            dicom_data = pydicom.dcmread(file_path)
            metadata = self.dicom_to_dict(dicom_data, file_path, type)
            metadata_list.append(metadata)
        metadata_df = pd.DataFrame(metadata_list)
        return metadata_df

    def dicom_to_dict(self, dicom, file_path, type=None):
        """ Parses the DICOM Data and extracts relevant fields
        into a dictionary. """
        
        data = {}

        data['patient_name'] = dicom.PatientName
        data['patient_id'] = dicom.PatientID
        data['patient_sex'] = dicom.PatientSex
        data['patient_age'] = int(dicom.PatientAge)
        data['pixel_spacing'] = dicom.PixelSpacing
        data['file_path'] = file_path
        data['id'] = dicom.SOPInstanceUID

        # Training data has labelled annotations
        if type == "train":
            encoded_pixels_list = self.rle[self.rle['ImageId']==dicom.SOPInstanceUID]['EncodedPixels'].values
            # If there are any annotations for pneumothorax in the data
            # mark the presense of pneumothorax as positive (True)
            pneumothorax = False
            for encoded_pixels in encoded_pixels_list:
                if encoded_pixels != '-1':
                    pneumothorax = True
            # get meaningful information (for train set)
            data['encoded_pixels_list'] = encoded_pixels_list
            data['pneumothorax'] = pneumothorax
            data['encoded_pixels_count'] = len(encoded_pixels_list)

        return data
    
    
class DataGenerator():
    """ Generator Class
    Will supply data lazily for ML Inference.
    """
    
    def __init__(self, train_data):
        self.train_data = train_data
        # Generate Masks for use in Data Generator
        self.masks = {}
        for index, row in self.train_data[self.train_data['pneumothorax']==1].iterrows():
            self.masks[row['id']] = list(row['encoded_pixels_list'])
        self.selectTrainingData()
        
    def selectTrainingData(self, num_negative_drop=0):
        # Remove data with 0 encoded pixel count -> x-rays with missing labels
        drop_data = self.train_data[self.train_data['encoded_pixels_count']==0].index
        self.selected_train_data = self.train_data.drop(drop_data)
        # Remove 'n' samples from the Negative Class as it outnumbers the positive Class
        drop_data = self.selected_train_data[self.selected_train_data['pneumothorax']==False].sample(num_negative_drop).index
        self.selected_train_data = self.selected_train_data.drop(drop_data)
        
    # def splitSelectedData(self, val_size=0.2):
    #     """ Split dataset for training and validation """
    #     # Split Dataset
    #     self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.selected_train_data.index, self.selected_train_data['pneumothorax'].values, test_size=val_size, random_state=42)
    #     self.X_train, self.X_val = self.selected_train_data.loc[self.X_train]['file_path'].values, self.selected_train_data.loc[self.X_val]['file_path'].values
        
    def prepareImages(self, file_list):
        """ Takes list of files and returns preprocessed images/labels for training """
        X = np.empty((self.batch_size, self.img_size, self.img_size, self.channels))
        y = np.empty((self.batch_size, self.img_size, self.img_size, self.channels))
        for idx, file_path in enumerate(file_list):
            id = file_path.split('/')[-1][:-4]
            rle = self.masks.get(id)
            image = pydicom.read_file(file_path).pixel_array
            image_resized = cv2.resize(image, (self.img_size, self.img_size))
            image_resized = np.array(image_resized, dtype=np.float64)
            X[idx, ] = np.expand_dims(image_resized, axis=2)
            # if there is no mask create empty mask
            # notice we are starting of with 1024 because we need to use the rle2mask function
            if rle is None:
                mask = np.zeros((1024, 1024))
            else:
                if len(rle) == 1:
                    mask = rle2mask(rle[0], 1024, 1024).T
                else: 
                    mask = np.zeros((1024, 1024))
                    # Take all masks / annotations into a single output
                    for r in rle:
                        mask =  mask + rle2mask(r, 1024, 1024).T
            
            mask_resized = cv2.resize(mask, (self.img_size, self.img_size))
            y[idx,] = np.expand_dims(mask_resized, axis=2)
        
        # Normalise
        X = X/255
        y = (y > 0).astype(int)
        return X, y
        
    def generateBatches(self, fns, batch_size=32, img_size=512, channels=1, shuffle=True):
        """ Generate Data to Supply to Fit Function """
        
        # self.selected_train_data['file_path']
        # fns = data['file_path'].values
        if shuffle is True:
            random.shuffle(fns)
            
        # Data Generator configurations
        self.channels = channels
        self.img_size = img_size
        self.batch_size = batch_size
            
        total_data = len(fns)
        batches = 0
        index = 0
        
        c.debugPrint("Length Total Data: {}".format(total_data), 1)
        while (batches < total_data//self.batch_size):
            c.debugPrint("Index Range: {}:{}".format(index, index+self.batch_size), 2)
            yield self.prepareImages(fns[index:index+self.batch_size])
            index += self.batch_size
            batches += 1