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
dc = dh.DataController()
#%% Inference Controller
ic = ih.InferenceController()
#%% Plotting Controller
pc = ph.PlottingController()
#%% Visualisations 
pc.plot_train_data(dc.train_data)
pc.plot_train_data_with_box(dc.train_data)
#%% Data Analysis
pc.data_analysis(dc.train_data)
