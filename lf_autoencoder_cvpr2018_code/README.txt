Our project consist of 3 steps:

1. Divide input light fields into 3D patches and create network inputs
2. Train the network
3. Run the network on test data

##########Input data#############
If you want to use our data then just specify the path to the data containers in "config_autoencoder_v9_final.py"
If you want to generate your own  data, please see ReadData project for the example.

##########Newtork training############
To train the network you need to specify all training options in the "config_autoencoder_v9_final.py"
Also, you need to specify patch size and minimum and maximum disparity values in the "config_data_format.py"
In "cnn_autoencoder.py" you need to specify coordinates that are taken into account when the loss is computed.
For example, if the input patch size is 48x48, and we select loss_min_coord_3D= 0, loss_max_coord_3D = 40,
then the last 8 pixels will be omitted while computing loss.

#########Network evaluation###########
One the training is completed, please use "test_autoencoder_v9_interp.py" for the evaluations.
If you want to use our data for the evaluations please make sure that all test data is in "test_data" folder, otherwise
see how to generate test data in the ReadData project.
In "config_autoencoder_v9_final.py" please specify the interpolation mask parameters, it is needed to assemble the whole light field from the
horizontal and vertical patches.
Specify datasets that you want to evaluate in "data_folders" and run the code.
Make sure that you import the correct config file.
Results will be saved in "results" folder.

For the convenience, we will have separate archives for the train and test data. Please make sure that you specified paths to the data.

If you have any questions, please contact us. We will be glad to help you :)







