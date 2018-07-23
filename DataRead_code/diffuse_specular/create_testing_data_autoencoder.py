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

test_data_dir      = "D:\\Python\\DataRead\\testData\\max_plank\\not seen\\"
training_data_filename = 'lf_test_intrinsic'

data_source = "H:\\MATLAB_DEEP\\intrinsic_lightfields_EMMCVPR2017\\Data\\max_plank\\guitar"
data_folders = os.listdir(data_source)
# data_folders = data_folders[4:5]


for lf_name in data_folders:

  file = h5py.File(test_data_dir + training_data_filename + lf_name + '.hdf5', 'w')
  data_folder = os.path.join(data_source,lf_name)
  # read diffuse color
  LF_dc = file_io.read_lightfield_intrinsic(data_folder, 'dc')
  # read diffuse direct
  LF_dd = file_io.read_lightfield_intrinsic(data_folder, 'dd')
  #read diffuse indirect
  LF_di = file_io.read_lightfield_intrinsic(data_folder, 'di')
  # read glossy color
  LF_gc = file_io.read_lightfield_intrinsic(data_folder, 'gc')
  # read glossy direct
  LF_gd = file_io.read_lightfield_intrinsic(data_folder, 'gd')
  #read glossy indirect
  LF_gi = file_io.read_lightfield_intrinsic(data_folder, 'gi')

  # diffuse LF
  LF_diffuse = np.multiply(LF_dc, np.add(LF_dd, LF_di))
  LF_specular = np.multiply(LF_gc, np.add(LF_gd, LF_gi))
  LF = np.add(LF_diffuse, LF_specular)
  cv_gt = lf_tools.cv(LF)

  # imean = 0.3
  # factor = imean / np.mean(cv_gt)
  # LF_diffuse = LF_diffuse*factor
  # LF_specular = LF_specular*factor
  # LF = np.add(LF_diffuse, LF_specular)


  dset_LF_diffuse = file.create_dataset('LF_diffuse', data=LF_diffuse)
  # glossy LF

  dset_LF_specular = file.create_dataset('LF_specular', data=LF_specular)
  # input LF

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
    
