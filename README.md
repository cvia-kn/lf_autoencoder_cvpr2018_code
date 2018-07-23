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

## Project description
Our project consist of 2 steps:

1. Divide input light fields into 3D patches and create network inputs with DataRead project
2. Train and evaluate the network with lf_autoencoder_cvpr2018_code project

### Prerequisites
1. Pythom 3.5
2. Tensorflow with GPU support

### 1. Creating the data
Depends on the type of data use separate scripts to create inputs for the network: 
* synthetic **create_training_data_autoencoder.py**
* [light field benchmark](http://hci-lightfield.iwr.uni-heidelberg.de/) **create_training_data_depth.py**
* real-world use separete script **create_training_data_lytro**
```
px = 48 # patch size
py = 48 
nviews = 9 # number of views
sx = 16 # block step size
sy = 16

training_data_dir = "H:\\trainData\\"
training_data_filename = 'lf_patch_autoencoder1.hdf5'
file = h5py.File( training_data_dir + training_data_filename, 'w' )

data_source = "H:\\CNN_data\\1"
```
