#!/usr/bin/python3
#
# read a bunch of source light fields and write out
# training data for our autoencoder in useful chunks
#
# pre-preparation is necessary as the training data
# will be fed to the trainer in random order, and keeping
# several light fields in memory is impractical.
#
# WARNING: store data on an SSD drive, otherwise randomly
# assembing a bunch of patches for training will
# take ages.
#
# (c) Bastian Goldluecke, Uni Konstanz
# bastian.goldluecke@uni.kn
# License: Creative Commons CC BY-SA 4.0
#

from queue import Queue
import time
import code
import os
import sys
import h5py

import numpy as np
# import deepdish as dd




# python tools for our lf database
import file_io
# additional light field tools
import lf_tools



# OUTPUT CONFIGURATION

# patch size. patches of this size will be extracted and stored
# must remain fixed, hard-coded in NN
px = 96
py = 96

# number of views in H/V/ direction
# input data must match this.
nviews = 9

# block step size. this is only 16, as we keep only the center 16x16 block
# of each decoded patch (reason: reconstruction quality will probably strongly
# degrade towards the boundaries).
# 
# TODO: test whether the block step can be decreased during decoding for speedup.
#
sx = 32
sy = 32

# output file to write to
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!! careful: overwrite mode !!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
# previous training data will be erased.

test_data_dir      = "H:\\testData\\diffuse_specular\\lytro\\not seen\\"
training_data_filename = 'lf_test_lytro'

# data_source = "H:\\MATLAB_DEEP\\intrinsic_lightfields_EMMCVPR2017\\Data\\max_plank\\guitar"
data_source = "E:\\LYTRODATA\\tea_bowl\\"
data_folders = os.listdir(data_source)
# data_folders = data_folders[16:17]


for lf_name in data_folders:

  file = h5py.File(test_data_dir + training_data_filename + '_' + lf_name + '.hdf5', 'w')
  data_file = os.path.join(data_source,lf_name)
  mat_content = h5py.File(data_file, 'r')
  LF = mat_content['LF'].value
  LF = np.transpose(LF, (4, 3, 2, 1, 0))

  dset_LF = file.create_dataset('LF', data = LF)

  # next dataset
  print(' done.')
    
