"""
Created on Mon Aug 11 00:07:13 2020
Testing Helper Functions and Classes

@author: James Sorrell

"""

import os
import json
import pydicom
import cv2
# Imports for Testing
import tensorflow as tf
import numpy as np
import pandas as pd
from skimage import morphology
import matplotlib.pyplot as plt
import config as c

# Import Mask Functions from Provided Files
# TODO: It would be cleaner to do this as a 
# module import, but I need to get that working
# cleanly with Syder, for now this will do
import sys
sys.path.insert(0, '../input')
from mask_functions import rle2mask, mask2rle

def plot_train(img, mask, pred, save_path=None):
    """ Take an image, mask and predicted mask and plot them for inspection """
    
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15,5))
    
    ax[0].imshow(img, cmap=plt.cm.bone)
    ax[0].set_title('Chest X-Ray')
    
    ax[1].imshow(mask, cmap=plt.cm.bone)
    ax[1].set_title('Mask')
    
    ax[2].imshow(pred, cmap=plt.cm.bone)
    ax[2].set_title('Pred Mask')
    
    c.debugPrint("Saving to: {}".format(save_path), 2)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    plt.close()

def remove_small_regions(img, size):
    """Morphologically removes small connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img

def prediction_post_processing(pred, img_size):
    """ Post processing pipeline for predictions """
    pred = (pred > .5).astype(int)
    return remove_small_regions(pred, 0.02 * np.prod(1024))

def plot_results(model_path, generator, img_size, save_path=None):
    """ Load model from path and test it """
    model = tf.keras.models.load_model(model_path, compile=False)
    # lets loop over the predictions and print some good-ish results
    count = 0
    savedImages = 0
    model_dir = os.path.dirname(model_path)
    c.createFolder(os.path.join(model_dir, "images"))
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
                    if save_path is not None and savedImages < 50:
                        # Plots are overwriting each other, needs to be done better
                        # but don't want to save all the plots into the repo
                        save_path = os.path.join(model_dir, "images", "{}.png".format(savedImages))
                        savedImages += 1
                    plot_train(img, mask, pred, save_path)
                    count += 1

def dice_coefficient(true, pred):
    """ Simple function to calculate dice loss coefficient """
    # Dice Coefficient is 2 * the Area of Overlap divided by the total number of pixels in both images.
    true = np.asarray(true).astype(np.bool)
    pred = np.asarray(pred).astype(np.bool)
    if not (np.sum(true) + np.sum(pred)):
        return 1.0
    # NOTE: Could add smoothing
    return (2. * np.sum(true * pred)) / (np.sum(true) + np.sum(pred))

def analyse_model(model_path, generator, img_size):
    """ Generate some metrics for model performance """
    model = tf.keras.models.load_model(model_path, compile=False)
    model_dir = os.path.dirname(model_path)
    p_dice_sum, p_count, dice_sum, count, correct, false_positive, false_negative = 0, 0, 0, 0, 0, 0, 0
    for x, y in generator:
        predictions = model.predict(x)
        for idx, val in enumerate(x):
            mask = np.reshape(y[idx]* 255, (img_size, img_size))
            diagnosis = np.any(mask)
            pred = np.reshape(predictions[idx], (img_size, img_size))
            pred = prediction_post_processing(pred, img_size)
            pred_diagnosis = np.any(pred)
            dice_coef = dice_coefficient(mask, pred)
            if (diagnosis == pred_diagnosis ):
                correct += 1
            elif (diagnosis == True and pred_diagnosis == False):
                false_negative += 1
            elif (diagnosis == False and pred_diagnosis == True):
                false_positive += 1
            if (diagnosis == True):
                p_dice_sum += dice_coef
                p_count += 1
            dice_sum += dice_coef
            count += 1
    c.debugPrint("\nMean Dice Coefficient: {:.2f} from {} test samples.".format(dice_sum/count, count), 0)
    c.debugPrint("Mean Positive Dice Coeff: {:.2f}".format(p_dice_sum/p_count), 0)
    correct_p = 100*(correct/count)
    incorrect_p = 100*((false_negative+false_positive)/count)
    fp_p = 100*(false_positive/count)
    fn_p = 100*(false_negative/count)
    c.debugPrint("Correct: {:.2f}, Incorrect: {:.2f}\nFalse Positive: {:.2f}, False Negative: {:.2f}".format(correct_p, incorrect_p, fp_p, fn_p), 0)
    results = {}
    results['total_tested'] = count
    results['correct'] = correct
    results['correct_percentage'] = correct_p
    results['false_positive_percentage'] = fp_p
    results['false_negative_percentage'] = fn_p
    results['mean_dice_coefficient'] = dice_sum/count
    results['mean_positive_dice_coefficient'] = p_dice_sum/p_count
    results['positive_count'] = p_count
    results_loc = os.path.join(model_dir, 'results.json')
    with open(results_loc, 'w') as fp:
        json.dump(results, fp)

def get_test_tensor(file_path, batch_size, img_size, channels):
        """ Takes filepath from test dataset and generates batch for processing """
        X = np.empty((batch_size, img_size, img_size, channels))
        # Store sample
        pixel_array = pydicom.read_file(file_path).pixel_array
        image_resized = cv2.resize(pixel_array, (img_size, img_size))
        image_resized = np.array(image_resized, dtype=np.float64)
        image_resized -= image_resized.mean()
        image_resized /= image_resized.std()
        X[0,] = np.expand_dims(image_resized, axis=2)
        return X

def prepare_submission(model_path, test_data, img_size):
    """ Prepare submission from provided model """
    # Save location
    model = tf.keras.models.load_model(model_path, compile=False)
    model_dir = os.path.dirname(model_path)
    results_loc = os.path.join(model_dir, 'submission.csv')
    # Prepare Submission
    submission = []
    print("Test Data: {}".format(len(test_data)))
    for i, row in test_data.iterrows():
        test_img = get_test_tensor(test_data['file_path'][i],1,img_size,1)
        # Get prediction
        pred_mask = model.predict(test_img).reshape((img_size, img_size))
        prediction = {}
        prediction['ImageId'] = str(test_data['id'][i])
        # Resize predicted mask
        pred_mask = cv2.resize(pred_mask.astype('float32'), (1024, 1024))
        pred_mask = prediction_post_processing(pred_mask, 1024)
        if pred_mask.sum() < 1:
            prediction['EncodedPixels']=  -1
        else:
            prediction['EncodedPixels'] = mask2rle(pred_mask.T * 255, 1024, 1024)
        submission.append(prediction)
    print("Submission Prepared!")
    # submission to csv
    submission_df = pd.DataFrame(submission)
    submission_df = submission_df[['ImageId','EncodedPixels']]
    # check out some predictions and see if it looks good
    submission_df[ submission_df['EncodedPixels'] != -1].head()
    submission_df.to_csv(results_loc, index=False)