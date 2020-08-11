import argparse
import inferenceHelpers as ih
import dataHelpers as dh
import plottingHelpers as ph
import testingHelpers as th

def main(model_path):
    """ Main Function
    Core Pipeline for Program Execution
    """

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
    training_dg = dh.DataGenerator(dc.train_data, img_size)
    #%%
    # Take small % of the data for Visualisation Purposes
    # For the best possible model we shouldn't do this
    # as we likely want to make use of all of the data
    X_train, X_val = training_dg.splitSelectedData(0.01)
    #%%
    if model_path is None:
        # If we specified a path skip model generation
        for X, y in training_dg.generateBatches(X_train):
            print("X: {}".format(X.shape))
            print("y: {}".format(y.shape))
            print("X type: {}".format(X.dtype))
            print("Y type: {}\n".format(y.dtype))
            break
        #%%
        ic = ih.InferenceController(img_size=img_size)
        #%%
        dataset = X_train
        #dataset = training_dg.selected_data['file_path'].values 
        print("Training Data Length: {}".format(len(dataset)))
        epochs = 100
        augmentation_factor = 10
        steps_per_epoch = (len(dataset)*augmentation_factor) // (training_dg.batch_size*epochs)
        print("Steps per epoch: {}".format(steps_per_epoch))
        generator = training_dg.generateBatches(dataset, augmentation_factor)
        #%%
        model_path = ic.train(generator, epochs, steps_per_epoch)
    #%%
    test_dataset = X_val
    #%%
    # Plotting some of the results from test dataset
    # th.plot_results(model_path, training_dg.generateBatches(test_dataset), img_size)
    #%%
    # Analyse Results
    th.analyse_model(model_path, training_dg.generateBatches(test_dataset), img_size)
    #%%
    # Create Submission


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Benchmarking Tool')
    parser.add_argument('path', default=None, type=str, help='path to model')
    options = parser.parse_args()
    main(options.path)