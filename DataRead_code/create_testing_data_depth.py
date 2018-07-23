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
px = 48
py = 48

# number of views in H/V/ direction
# input data must match this.
nviews = 9

# block step size. this is only 16, as we keep only the center 16x16 block
# of each decoded patch (reason: reconstruction quality will probably strongly
# degrade towards the boundaries).
# 
# TODO: test whether the block step can be decreased during decoding for speedup.
#
sx = 16
sy = 16

# output file to write to
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!! careful: overwrite mode !!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
# previous training data will be erased.

test_data_dir      = "H:\\testData\\diffuse_specular\\benchmark\\seen\\"
training_data_filename = 'lf_benchmark_'

data_source = "E:\\MATLAB\\Benchmark_dataset\\data"
# data_folders = os.listdir(data_source)
data_folders = []
data_folders.append('antinous')
data_folders.append('greek')
data_folders.append('vinyl')


for lf_name in data_folders:

  file = h5py.File(test_data_dir + training_data_filename + lf_name + '.hdf5', 'w')
  data_folder = os.path.join(data_source,lf_name)
  LF = file_io.read_lightfield(data_folder)
  LF = LF.astype(np.float32) / 255.0
  dset_LF = file.create_dataset('LF', data = LF)

  disp = file_io.read_disparity( data_folder )
  disp_gt = np.array( disp[0] )
  disp_gt = np.flip( disp_gt,0 )
  dset_disp = file.create_dataset('LF_disp', data=disp_gt)

  # maybe we need those, probably not.
  # param_dict = file_io.read_parameters(data_folder)
  # dd.io.save(file, param_dict)

  # next dataset
  print(' done.')
    
