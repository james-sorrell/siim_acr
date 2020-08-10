"""
Created on Mon Aug 11 00:07:13 2020
Testing Helper Functions and Classes

@author: James Sorrell

"""

# Imports for Testing
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def plot_train(img, mask, pred):
    """ Take an image, mask and predicted mask and plot them for inspection """
    
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15,5))
    
    ax[0].imshow(img, cmap=plt.cm.bone)
    ax[0].set_title('Chest X-Ray')
    
    ax[1].imshow(mask, cmap=plt.cm.bone)
    ax[1].set_title('Mask')
    
    ax[2].imshow(pred, cmap=plt.cm.bone)
    ax[2].set_title('Pred Mask')
    
    plt.show()

def analyse_data(model_path, generator, img_size):
    """ Load model from path and test it """
    model = tf.keras.models.load_model(model_path, compile=False)
    # lets loop over the predictions and print some good-ish results
    count = 0
    for i in range(0,50):
        if count <= 50:
            x, y = next(generator)
            predictions = model.predict(x)
            for idx, val in enumerate(x):
                #if y[idx].sum() > 0 and count <= 15: 
                    img = np.reshape(x[idx]* 255, (img_size, img_size))
                    mask = np.reshape(y[idx]* 255, (img_size, img_size))
                    pred = np.reshape(predictions[idx], (img_size, img_size))
                    pred = pred > 0.5
                    pred = pred * 255
                    plot_train(img, mask, pred)
                    count += 1