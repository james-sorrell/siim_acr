import inferenceHelpers as ih
import dataHelpers as dh
import plottingHelpers as ph

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
pc.plot_train_data(dc.train_data)
pc.plot_train_data_with_box(dc.train_data)
#%% Data Analysis
pc.data_analysis(dc.train_data)
#%%
dg = dh.DataGenerator(dc.train_data)
#%%
# dg.splitSelectedData()
#%%
# dg.generateBatches(dg.X_train)
# dg.generateBatches(dg.X_val)
#%%
for X, y in dg.generateBatches(dg.selected_train_data['file_path'].values):
    print("X: {}".format(X.shape))
    print("y: {}".format(y.shape))
    print("X type: {}".format(X.dtype))
    print("Y type: {}".format(y.dtype))
    break
#%%
# for X, y in dg.generateBatches(dg.X_val):
#     print("X: {}".format(X.shape))
#     print("y: {}".format(y.shape))
#     break
#%% Inference Controller
ic = ih.InferenceController()
#%%
print("Training Data Length: {}".format(len(dg.selected_train_data)))
steps_per_epoch = len(dg.selected_train_data) // dg.batch_size
#%%
generator = dg.generateBatches(dg.selected_train_data['file_path'].values)
#%%
#ic.train(generator, steps_per_epoch)