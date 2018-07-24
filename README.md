# Light field intrinsics with a deep encoder-decoder network

We present a fully convolutional autoencoder for light fields,
which jointly encodes stacks of horizontal and vertical
epipolar plane images through a deep network of residual layers.
The complex structure of the light field is thus reduced to
a comparatively low-dimensional representation, which can be decoded
in a variety of ways.
The different pathways of upconvolution we currently support
are for disparity estimation and separation of the lightfield
into diffuse and specular intrinsic components.
The key idea is that we can jointly perform unsupervised training
for the autoencoder path of the network, and supervised training
for the other decoders. This way, we find features which are
both tailored to the respective tasks and generalize
well to datasets for which only example light fields
are available.
We provide an extensive evaluation on synthetic light field
data, and show that the network yields good results
on previously unseen real world data captured by a Lytro Illum camera
and various gantries.
![teaser](https://user-images.githubusercontent.com/41570345/43076792-9e4eaf92-8e85-11e8-8763-9e3fd041ab23.png)

## Project description
Our project consist of 2 steps:

1. Divide input light fields into 3D patches and create network inputs with [DataRead_code](https://github.com/cvia-kn/lf_autoencoder_cvpr2018_code/tree/master/DataRead_code) project
2. Train and evaluate the network with [lf_autoencoder_cvpr2018_code](https://github.com/cvia-kn/lf_autoencoder_cvpr2018_code/tree/master/lf_autoencoder_cvpr2018_code) project

### Prerequisites
1. Python 3.5
2. Tensorflow with GPU support

### 1. Creating the data
Depends on the type of data use separate scripts to create inputs (.hdf5 data container) for the network: 
* synthetic **create_training_data_autoencoder.py**
* [light field benchmark](http://hci-lightfield.iwr.uni-heidelberg.de/) **create_training_data_depth.py**
* real-world use separete script **create_training_data_lytro**
```
px = 48 # patch size
py = 48 
nviews = 9 # number of views
sx = 16 # block step size
sy = 16

training_data_dir = "./trainData/"
training_data_filename = 'lf_patch_autoencoder1.hdf5'
file = h5py.File( training_data_dir + training_data_filename, 'w' )

data_source = "./CNN_data/1"
```
[Synthetic data](http://link/) used for training, is organized in 8 folders, for every folder we create a data container.

### 2. Run the network
To train the network you need to specify all training options in the **config_autoencoder_v9_final.py**
Also, you need to specify patch size and minimum and maximum disparity values in the **config_data_format.py**
In **cnn_autoencoder.py** you need to specify coordinates that are taken into account when the loss is computed.
For example, if the input patch size is 48x48, and we select *loss_min_coord_3D= 0*, *loss_max_coord_3D = 40*,
then the last 8 pixels will be omitted while computing loss.

To use the trained model, please download the model [current_v9.zip](http://data.lightfield-analysis.net/current_v9.zip) and extract the archive to **./networks/** folder.
We provide some test examples [diffuse_specular.zip](http://data.lightfield-analysis.net/diffuse_specular.zip) that shoulb be extracted to the **./test_data/** folder.

To evaluate on all test examples used in the paper, create the test data with the [DataRead_code](https://github.com/cvia-kn/lf_autoencoder_cvpr2018_code/tree/master/DataRead_code) project.

### References
* [Our webpage](https://www.cvia.uni-konstanz.de/)
* [Paper](http://publications.lightfield-analysis.net/AJSG18_cvpr.pdf)
* [Supplementary material](http://publications.lightfield-analysis.net/AJSG18_cvpr_supplemental.pdf)

