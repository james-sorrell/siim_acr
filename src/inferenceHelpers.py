"""
Created on Mon Aug 10 15:39:23 2020
Inference Helper Functions and Classes to process SIIM ACR Data

@author: James Sorrell

"""

# Imports for Inference
import cv2
import tensorflow as tf
from skimage import morphology, io, color, exposure, img_as_float, transform

class InferenceController():
    """ 
    Inference Controller Class
    Class that handles Inference and Preprocessing
    """

    def __init__(self):
        print("TBD")