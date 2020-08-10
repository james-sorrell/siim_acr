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

import pydicom
import config as c
from glob import glob
from tqdm import tqdm
import pandas as pd

class DataController():
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
        train_fns = sorted(glob(path))
        metadata_df = pd.DataFrame()
        metadata_list = []
        for file_path in tqdm(train_fns):
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