"""
Created on Mon Aug 10 15:09:37 2020
Data Helper Functions and Classes to process SIIM ACR Data

https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/data

The data is comprised of images in DICOM format and annotations in the form of image IDs 
and run-length-encoded (RLE) masks. Some of the images contain instances of pneumothorax 
(collapsed lung), which are indicated by encoded binary masks in the annotations. 
Some training images have multiple annotations.

@author: James Sorrell

Data Loader Inspired By:
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
from matplotlib.colors import rgb_to_hsv

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
    
    def __init__(self, data, img_size, batch_size=16, channels=1, crop_data=True):
        # Data Generator configurations
        self.data = data
        self.img_size = img_size
        self.channels = channels
        self.batch_size = batch_size
        # Generate Masks for use in Data Generator
        self.masks = {}
        for index, row in self.data[self.data['pneumothorax']==True].iterrows():
            self.masks[row['id']] = list(row['encoded_pixels_list'])
        if crop_data is True:   
            self.selectData()

    def selectData(self, negative_ratio=1):
        """ Configurations for what data should be removed from the dataset 
            Negative ratio defined to be 1:x Positive to Negative Samples
        """
        # Remove data with 0 encoded pixel count -> x-rays with missing labels
        drop_data = self.data[self.data['encoded_pixels_count']==0].index
        self.selected_data = self.data.drop(drop_data)
        # Drop samples such that we have a 50% positive and 50% negative samples
        num_positive_samples = len(self.selected_data[self.selected_data['pneumothorax']==True])
        num_negative_allowed = negative_ratio*num_positive_samples
        num_negative_samples = len(self.selected_data[self.selected_data['pneumothorax']==False])
        num_negative_drop = num_positive_samples - num_negative_allowed
        if num_negative_drop < 0:
            num_negative_drop = 0
        # num_negative_drop -= len(self.selected_data[self.selected_data['pneumothorax']==True])
        c.debugPrint("\nDropping {} of the Negative Class!\n".format(num_negative_drop), 1)
        # Remove 'n' samples from the Negative Class as it outnumbers the positive Class
        drop_data = self.selected_data[self.selected_data['pneumothorax']==False].sample(num_negative_drop).index
        self.selected_data = self.selected_data.drop(drop_data)

    def chance(self, chance):
        """ Percentage chance to return true """
        eps = chance/100
        if (random.random() < eps):
            return True
        return False

    def change_brightness(self, img, value):
        """ Change brightness of an image by value """
        img += value
        img[img>255] = 255
        img[img<0] = 0
        return img

    def rotateHelper(self, img, angle):
        """ Rotate an Arr by provided angle """
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
        return cv2.warpAffine(img, M, (w, h))

    def rotate(self, img, mask, angle):
        """ Rotate Image and Mask by provided Angle """
        img = self.rotateHelper(img, angle)
        mask = self.rotateHelper(mask, angle)
        return img, mask

    def horizontal_flip(self, img, mask):
        """ Flip image and mask horizontally """
        return cv2.flip(img, 1), cv2.flip(mask, 1)

    def vertical_flip(self, img, mask):
        """ Flip image and mask horizontally """
        return cv2.flip(img, 0), cv2.flip(mask, 0)

    def shiftHelper(self, img, ox, oy):
        """ Shifting operation """
        non = lambda s: s if s<0 else None
        mom = lambda s: max(0,s)
        new_img = np.zeros(img.shape)
        new_img[mom(oy):non(oy), mom(ox):non(ox)] = img[mom(-oy):non(-oy), mom(-ox):non(-ox)]
        return new_img

    def shift(self, img, mask, ox, oy):
        """ Shift image and mask by specified values """
        return self.shiftHelper(img, ox, oy), self.shiftHelper(mask, ox, oy)

    def augmentData(self, img, mask):
        """ Provide a set of augmentations to the dataset, this should help with 
        algorithm robustness """
        # brightness
        if self.chance(80):
            img = self.change_brightness(img, value=random.randint(-60,100))
        # flip horizontal
        if self.chance(50):
            img, mask = self.horizontal_flip(img, mask)
        # flip vertical
        if self.chance(50):
            img, mask = self.vertical_flip(img, mask)
        # rotation
        if self.chance(75):
            angle = int(random.uniform(-40, 40))
            img, mask = self.rotate(img, mask, angle)
        # translation
        if self.chance(40):
            ox = random.randint(-40,40)
            oy = random.randint(-40,40)
            img, mask = self.shift(img, mask, ox, oy)
        return img, mask

    def splitSelectedData(self, val_size=0.2):
        """ Split dataset for training and validation """
        # Shuffle first
        self.selected_data = self.selected_data.sample(frac=1).reset_index(drop=True)
        # Split Dataset
        X_train, X_val, _, _ = train_test_split(self.selected_data.index, self.selected_data['pneumothorax'].values, test_size=val_size, random_state=42)
        return self.selected_data.loc[X_train]['file_path'].values, self.selected_data.loc[X_val]['file_path'].values
        
    def prepareImages(self, file_list):
        """ Takes list of files and returns preprocessed images/labels for training """
        X = np.empty((self.batch_size, self.img_size, self.img_size, self.channels))
        y = np.empty((self.batch_size, self.img_size, self.img_size, self.channels))
        for idx, file_path in enumerate(file_list):
            id = file_path.split('\\')[-1][:-4]
            rle = self.masks.get(id)
            image = pydicom.read_file(file_path).pixel_array
            image_resized = cv2.resize(image, (self.img_size, self.img_size))
            image_resized = np.array(image_resized, dtype=np.float64)
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
                        mask += rle2mask(r, 1024, 1024).T
            
            mask_resized = cv2.resize(mask, (self.img_size, self.img_size))
            image_resized, mask_resized = self.augmentData(image_resized, mask_resized)
            X[idx, ] = np.expand_dims(image_resized, axis=2)
            y[idx,] = np.expand_dims(mask_resized, axis=2)
        # Normalise
        X = X/255
        y = (y>0).astype(np.float64)
        return X, y
        
    def generateBatches(self, fns, augmentation_factor=1, shuffle=True):
        """ Generate Data to Supply to Fit Function """
        
        # self.selected_train_data['file_path']
        # fns = data['file_path'].values
        if shuffle is True:
            random.shuffle(fns)
            
        total_data = len(fns)
        c.debugPrint("\nLength Total Data: {}".format(total_data), 1)
        c.debugPrint("Augmentation Factor: {}\n".format(augmentation_factor), 1)
        for _ in range(augmentation_factor):
            batches = 0
            index = 0
            while (batches < total_data//self.batch_size):
                c.debugPrint("Index Range: {}:{}".format(index, index+self.batch_size), 2)
                yield self.prepareImages(fns[index:index+self.batch_size])
                index += self.batch_size
                batches += 1