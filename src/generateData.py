# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 20:58:08 2020

@author: James Sorrell
"""


class DataGenerator():
    
    def __init__(self, train_data, batch_size=32, img_size=256, channels=1, shuffle=True):
        self.train_data = train_data
        # Data Generator configurations
        self.shuffle = shuffle
        self.channels = channels
        self.img_size = img_size
        self.batch_size = batch_size