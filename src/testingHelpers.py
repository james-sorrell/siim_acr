"""
Created on Mon Aug 11 00:07:13 2020
Testing Helper Functions and Classes

@author: James Sorrell

"""

import os
# Imports for Testing
import tensorflow as tf
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt

def plot_train(img, mask, pred, save_path=None):
    """ Take an image, mask and predicted mask and plot them for inspection """
    
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15,5))
    
    ax[0].imshow(img, cmap=plt.cm.bone)
    ax[0].set_title('Chest X-Ray')
    
    ax[1].imshow(mask, cmap=plt.cm.bone)
    ax[1].set_title('Mask')
    
    ax[2].imshow(pred, cmap=plt.cm.bone)
    ax[2].set_title('Pred Mask')
    
    if save_path is not None:
        plt.savefig(save_path)

def remove_small_regions(img, size):
    """Morphologically removes small connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img

def prediction_post_processing(pred, img_size):
    """ Post processing pipeline for predictions """
    pred = pred > 0.5
    return remove_small_regions(pred, 0.02*img_size)

def plot_results(model_path, generator, img_size, save_path=None):
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
                    pred = prediction_post_processing(pred, img_size)
                    # Scale for visualisation
                    pred = pred * 255
                    if save_path is not None:
                        # Plots are overwriting each other, needs to be done better
                        # but don't want to save all the plots into the repo
                        save_path = os.path.join(save_path, "images", "{}.png".format(idx))
                    plot_train(img, mask, pred, save_path)
                    count += 1

def dice_coefficient(true, pred):
    """ Simple function to calculate dice loss coefficient """
    # Dice Coefficient is 2 * the Area of Overlap divided by the total number of pixels in both images.
    return np.sum(pred[true==1])*2.0 / (np.sum(pred) + np.sum(true))

def analyse_model(model_path, generator, img_size):
    """ Generate some metrics for model performance """
    model = tf.keras.models.load_model(model_path, compile=False)
    model_dir = os.path.dirname(model_path)
    sum, count, correct, false_positive, false_negative = 0, 0, 0, 0, 0
    for x, y in generator:
        predictions = model.predict(x)
        for idx, val in enumerate(x):
            mask = np.reshape(y[idx]* 255, (img_size, img_size))
            diagnosis = np.any(mask)
            pred = np.reshape(predictions[idx], (img_size, img_size))
            pred_diagnosis = np.any(pred > 0.5)
            dice_coef = dice_coefficient(mask, pred)
            if (diagnosis == pred_diagnosis ):
                correct += 1
            elif (diagnosis == True and pred_diagnosis == False):
                false_negative += 1
            elif (diagnosis == False and pred_diagnosis == True):
                false_positive += 1
            sum += dice_coef
            count += 1
    print("\nMean Dice Coefficient: {:.2f} from {} test samples.".format(sum/count, count))
    correct_p = 100*(correct/count)
    incorrect_p = 100*((false_negative+false_positive)/count)
    fp_p = 100*(false_positive/count)
    fn_p = 100*(false_negative/count)
    print("Correct: {:.2f}, Incorrect: {:.2f}\nFalse Positive: {:.2f}, False Negative: {:.2f}".format(correct_p, incorrect_p, fp_p, fn_p))
    