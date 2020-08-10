import inferenceHelpers as ih
import dataHelpers as dh
import plottingHelpers as ph
import testingHelpers as th

""" Core Pipeline for Program Execution """
#%% Cells for Spyder Execution
#
#%% Data Controller
# Create the data controller
# this will also prepare the 
# training and test data.
dc = dh.DataLoader()
#%% Plotting Controller
pc = ph.PlottingController()
#%% Visualisations 
# pc.plot_train_data(dc.train_data)
# pc.plot_train_data_with_box(dc.train_data)
#%% Data Analysis
# pc.data_analysis(dc.train_data)
#%%
img_size = 256
dg = dh.DataGenerator(dc.train_data, img_size)
#%%
# Take 1% of the data for Visualisation Purposes
# For the best possible model we shouldn't do this
# as we likely want to make use of all of the data
X_train, X_val = dg.splitSelectedData(0.01)
#%%
for X, y in dg.generateBatches(dg.selected_train_data['file_path'].values):
    print("X: {}".format(X.shape))
    print("y: {}".format(y.shape))
    print("X type: {}".format(X.dtype))
    print("Y type: {}".format(y.dtype))
    break
#%%
#%% Inference Controller
ic = ih.InferenceController(img_size=img_size)
#%%
# dataset = dg.selected_train_data
dataset = X_train
print("Training Data Length: {}".format(len(dataset)))
epochs = 10
steps_per_epoch = len(dataset) // (dg.batch_size*epochs)
print("Steps per epoch: {}".format(steps_per_epoch))
#%%
generator = dg.generateBatches(dataset)
#%%
model_path = ic.train(generator, epochs, steps_per_epoch)
#%%
# Testing
th.analyse_data('../models/2020-11-08-00-05.h5', dg.generateBatches(X_val), img_size)