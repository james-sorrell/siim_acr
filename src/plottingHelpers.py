# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 16:09:37 2020
Plotting Helper Functions and Classes to process SIIM ACR Data

@author: James Sorrell

Plotting functions for visualisation and understanding purposes
Code inspired from:
    https://www.kaggle.com/ekhtiar/finding-pneumo-part-1-eda-and-unet

"""

import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as patches
import pydicom

# Import Mask Functions from Provided Files
# TODO: It would be cleaner to do this as a 
# module import, but I need to get that working
# cleanly with Syder, for now this will do
import sys
sys.path.insert(0, '../input')
from mask_functions import rle2mask, mask2rle

class PlottingController():
    """ 
    Plotting Controller Class
    Class that handles data for plotting of images, bounding boxes
    as well as analysis of data for plotting purposes.
    """
    
    def __init__(self):
        print("Plotting Helpers")
    
    def data_analysis(self, train_data):
        """ Analysing Data to get a better understanding of potential bias
        and to inform inference decisions.
        """
        
        plt.figure()
        plot_frame = pd.DataFrame({'Pneumothorax': train_data['pneumothorax']})
        plot_frame.apply(pd.value_counts).plot(kind='bar', subplots=True, \
                                                grid=True, legend=False)
        plt.title('Pneumothorax Diagnosis')
        plt.ylabel("Counts")
        plt.xlabel("Diagnosis ")
        
        plt.figure()
        sb.distplot(train_data['patient_age'], bins=range(0,110,2), kde=False, color='c')
        plt.title('Patient Age Histogram')
        plt.ylabel("Counts")
        plt.xlabel("Age")
        plt.show()
        
        plt.figure()
        pneumo_age = train_data[train_data['pneumothorax']==True]['patient_age'].values
        no_pneumo_age = train_data[train_data['pneumothorax']==False]['patient_age'].values
        plt.hist(pneumo_age, bins=range(0,110,2), label='Positive', alpha=0.5, color='r')
        plt.hist(no_pneumo_age, bins=range(0,110,2), label='Negative', alpha=0.5, color='k')   
        plt.axvline(x=np.mean(pneumo_age), linewidth='1', color='r', linestyle='--', label="Mean - Positive")
        plt.axvline(x=np.mean(no_pneumo_age), linewidth='1', color='k', linestyle='--', label="Mean - Negative")
        plt.legend(loc='best')
        plt.title('Diagnosis vs. Age Histogram')
        plt.xlabel('Diagnosis')
        plt.ylabel('Counts')
        plt.show()
        
        male_train_data = train_data[train_data['patient_sex']=='M']
        female_train_data = train_data[train_data['patient_sex']=='F']
        male_positive_count = len(male_train_data[male_train_data['pneumothorax']==True])
        female_positive_count = len(female_train_data[female_train_data['pneumothorax']==True])
        male_negative_count = len(male_train_data[male_train_data['pneumothorax']==False])
        female_negative_count = len(female_train_data[female_train_data['pneumothorax']==False])
        male_positive_percentage = 100*(male_positive_count/(male_positive_count+male_negative_count))
        female_positive_percentage = 100*(female_positive_count/(female_positive_count+female_negative_count))

        fig, ax = plt.subplots()
        
        ax.bar(['Men', 'Women'], [male_positive_count, female_positive_count], label="Positive", color='r', alpha=0.5)
        ax.bar(['Men', 'Women'], [male_negative_count, female_negative_count], 
                bottom=[male_positive_count, female_positive_count], label="Negative", color='k', alpha=0.5)
        ax.set_ylabel('Counts')
        ax.set_title('Diagnosis by Sex')
        ax.text(0, male_positive_count*0.8, "{:.2f}%".format(male_positive_percentage), horizontalalignment='center', fontsize=12)
        ax.text(1, female_positive_count*0.7, "{:.2f}%".format(female_positive_percentage), horizontalalignment='center', fontsize=12)
        ax.legend()
        plt.show()
        
        total_samples = len(train_data)
        print("Total Number of Training Samples: {}".format(total_samples))
        positive_count = train_data['pneumothorax'].sum()
        positive_percentage = 100*(positive_count/total_samples)
        print("Pneumothorax Positive count: {} -> {:.2f}%".format(positive_count, positive_percentage))
        missing = train_data[train_data['encoded_pixels_count']==0]['encoded_pixels_count'].count()
        missing_percentage = 100*(missing/positive_count)
        print("Number of x-rays with missing labels: {} -> {:.2f}%".format(missing, missing_percentage))
        negative_count = len(train_data)-positive_count
        negative_percentage = 100*(negative_count/total_samples)
        print("Pneumothorax Negative count: {} -> {:.2f}".format(negative_count, negative_percentage))

    
    def bounding_box(self, img):
        """ Returns max and min of mask to draw bounding box """
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax

    def plot_with_bounding_box(self, file_path, mask_encoded_list):
        """ Plot Image with Mask """
        # TODO: Unsure if it's better to move this to top of function
        # as it may not always be used
        import cv2
        pixel_array = pydicom.dcmread(file_path).pixel_array
        # Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        clahe_pixel_array = clahe.apply(pixel_array)
        # use the masking function to decode RLE
        mask_decoded_list = [rle2mask(mask_encoded, 1024, 1024).T \
                             for mask_encoded in mask_encoded_list]
    
        fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(20,10))
        # print out the xray
        ax[0].imshow(pixel_array, cmap=plt.cm.bone)
        # print the bounding box
        for mask_decoded in mask_decoded_list:
            # print out the annotated area
            ax[0].imshow(mask_decoded, alpha=0.3, cmap="Reds")
            rmin, rmax, cmin, cmax = self.bounding_box(mask_decoded)
            bbox = patches.Rectangle((cmin,rmin),cmax-cmin,rmax-rmin,linewidth=1,edgecolor='r',facecolor='none')
            ax[0].add_patch(bbox)
        ax[0].set_title('With Mask')
    
        # plot image with clahe processing with just bounding box and no mask
        ax[1].imshow(clahe_pixel_array, cmap=plt.cm.bone)
        for mask_decoded in mask_decoded_list:
            rmin, rmax, cmin, cmax = self.bounding_box(mask_decoded)
            bbox = patches.Rectangle((cmin,rmin),cmax-cmin,rmax-rmin,linewidth=1,edgecolor='r',facecolor='none')
            ax[1].add_patch(bbox)
        ax[1].set_title('Without Mask - Clahe')
    
        # plot plain xray with just bounding box and no mask
        ax[2].imshow(pixel_array, cmap=plt.cm.bone)
        for mask_decoded in mask_decoded_list:
            rmin, rmax, cmin, cmax = self.bounding_box(mask_decoded)
            bbox = patches.Rectangle((cmin,rmin),cmax-cmin,rmax-rmin,linewidth=1,edgecolor='r',facecolor='none')
            ax[2].add_patch(bbox)
        ax[2].set_title('Without Mask')
        plt.show()
        
    def plot_train_data(self, train_data):
        """ Randomly plots 4 images """
        num_img = 4
        subplot_count = 0
        fig, ax = plt.subplots(nrows=1, ncols=num_img, sharey=True, figsize=(num_img*10,10))
        for index, row in train_data.sample(n=num_img).iterrows():
            dataset = pydicom.dcmread(row['file_path'])
            ax[subplot_count].imshow(dataset.pixel_array, cmap=plt.cm.bone)
            # label the x-ray with information about the patient
            ax[subplot_count].text(0,0,'Age:{}, Sex: {}, Pneumothorax: {}'.format(row['patient_age'],row['patient_sex'],row['pneumothorax']),
                                size=26,color='white', backgroundcolor='black')
            subplot_count += 1
        plt.show()
    
    def plot_train_data_with_box(self, train_data):
        """ Randomly plot some training data with pneumothoroax 
        with bounding box and mask """
        train_metadata_sample = \
            train_data[train_data['pneumothorax']==True].sample(n=10)
        # plot ten xrays with and without mask
        for index, row in train_metadata_sample.iterrows():
            file_path = row['file_path']
            mask_encoded_list = row['encoded_pixels_list']
            print('image id: ' + row['id'])
            self.plot_with_bounding_box(file_path, mask_encoded_list)