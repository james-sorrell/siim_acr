"""
Created on Mon Aug 10 15:39:23 2020
Inference Helper Functions and Classes to process SIIM ACR Data

@author: James Sorrell

"""

# Imports for Inference
import datetime
import tensorflow as tf
from tensorflow import reduce_sum
from tensorflow.keras.backend import pow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Concatenate, Add, Flatten
from tensorflow.keras.losses import binary_crossentropy

class InferenceController():
    """ 
    Inference Controller Class
    Class that handles Inference and Preprocessing
    """

    def __init__(self, img_size, lr=0.01, eps=0.1):
        self.img_size = img_size
        self.ResUNet()
        adam = tf.keras.optimizers.Adam(lr=lr, epsilon=eps)
        self.model.compile(optimizer=adam, loss=self.bce_dice_loss, metrics=[self.dsc])
               
    def train(self, generator, epochs, steps_per_epoch):
        """ Run training and save the model to models folder """
        # running more epoch to see if we can get better results
        history = self.model.fit(generator, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1)
        timestr = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        path = '../models/{}/{}.h5'.format(timestr)
        self.model.save(path)
        return path
        
    def bn_act(self, x, act=True):
        """ batch normalization layer with an optinal activation layer """
        x = tf.keras.layers.BatchNormalization()(x)
        if act == True:
            x = tf.keras.layers.Activation('relu')(x)
        return x
    
    def conv_block(self, x, filters, kernel_size=3, padding='same', strides=1):
        """ convolutional layer which always uses the batch normalization layer """
        conv = self.bn_act(x)
        conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
        return conv

    def stem(self, x, filters, kernel_size=3, padding='same', strides=1):
        conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
        conv = self.conv_block(conv, filters, kernel_size, padding, strides)
        shortcut = Conv2D(filters, kernel_size=1, padding=padding, strides=strides)(x)
        shortcut = self.bn_act(shortcut, act=False)
        output = Add()([conv, shortcut])
        return output
    
    def residual_block(self, x, filters, kernel_size=3, padding='same', strides=1):
        res = self.conv_block(x, filters, kernel_size, padding, strides)
        res = self.conv_block(res, filters, kernel_size, padding, 1)
        shortcut = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
        shortcut = self.bn_act(shortcut, act=False)
        output = Add()([shortcut, res])
        return output
    
    def dsc(self, y_true, y_pred):
        smooth = 1.
        y_true_f = Flatten()(y_true)
        y_pred_f = Flatten()(y_pred)
        intersection = reduce_sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (reduce_sum(y_true_f) + reduce_sum(y_pred_f) + smooth)
        return score
    
    def dice_loss(self, y_true, y_pred):
        loss = 1 - self.dsc(y_true, y_pred)
        return loss
    
    def bce_dice_loss(self, y_true, y_pred):
        """
        This competition is evaluated on the mean Dice coefficient. 
        The Dice coefficient can be used to compare the pixel-wise 
        agreement between a predicted segmentation and its 
        corresponding ground truth. 
        # https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/overview/evaluation
        """
        loss = binary_crossentropy(y_true, y_pred) + self.dice_loss(y_true, y_pred)
        return loss

    def upsample_concat_block(self, x, xskip):
        u = UpSampling2D((2,2))(x)
        c = Concatenate()([u, xskip])
        return c

    def ResUNet(self):
        """ Define ResUNet from blocks """
        f = [16, 32, 64, 128, 256, 512, 1024, 2048] * 32
        inputs = Input((self.img_size, self.img_size, 1))
        
        ## Encoder
        e0 = inputs
        e1 = self.stem(e0, f[0])
        e2 = self.residual_block(e1, f[1], strides=2)
        e3 = self.residual_block(e2, f[2], strides=2)
        e4 = self.residual_block(e3, f[3], strides=2)
        e5 = self.residual_block(e4, f[4], strides=2)
        e6 = self.residual_block(e5, f[5], strides=2)
        e7 = self.residual_block(e6, f[6], strides=2)
        
        ## Bridge
        b0 = self.conv_block(e7, f[6], strides=1)
        b1 = self.conv_block(b0, f[6], strides=1)
        
        ## Decoder
        u1 = self.upsample_concat_block(b1, e6)
        d1 = self.residual_block(u1, f[6])
        
        u2 = self.upsample_concat_block(d1, e5)
        d2 = self.residual_block(u2, f[3])
        
        u3 = self.upsample_concat_block(d2, e4)
        d3 = self.residual_block(u3, f[2])
        
        u4 = self.upsample_concat_block(d3, e3)
        d4 = self.residual_block(u4, f[1])
        
        u5 = self.upsample_concat_block(d4, e2)
        d5 = self.residual_block(u5, f[1])
        
        u6 = self.upsample_concat_block(d5, e1)
        d6 = self.residual_block(u6, f[1])
        
        outputs = tf.keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d6)
        self.model = tf.keras.models.Model(inputs, outputs)