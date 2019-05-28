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

clip_min = 0
clip_max = 2

# output file to write to
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!! careful: overwrite mode !!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
# previous training data will be erased.

test_data_dir      = "D:\\Python\\DataRead\\testData\\intrinsic_images\\intrinsic\\not seen\\"
training_data_filename = 'lf_test_intrinsic'

data_source = "H:\\CNN_new_data_180202\\test"
data_folders = os.listdir(data_source)
# data_folders = data_folders[6:9]


for lf_name in data_folders:

  file = h5py.File(test_data_dir + training_data_filename + lf_name + '.hdf5', 'w')
  data_folder = os.path.join(data_source,lf_name)
  # read diffuse color
  LF_dc = file_io.read_lightfield_intrinsic_crosshair(data_folder, 'dc')
  # read diffuse direct
  LF_dd = file_io.read_lightfield_intrinsic_crosshair(data_folder, 'dd')
  #read diffuse indirect
  LF_di = file_io.read_lightfield_intrinsic_crosshair(data_folder, 'di')
  # read glossy color
  LF_gc = file_io.read_lightfield_intrinsic_crosshair(data_folder, 'gc')
  # read glossy direct
  LF_gd = file_io.read_lightfield_intrinsic_crosshair(data_folder, 'gd')
  #read glossy indirect
  LF_gi = file_io.read_lightfield_intrinsic_crosshair(data_folder, 'gi')

  # albedo LF
  LF_albedo = LF_dc
  # shading LF
  LF_sh = np.add(LF_dd, LF_di)
  # min_v = np.amin(LF_sh)
  max_v = np.amax(LF_sh)
  if max_v > 2:
    print('rescaling')
    LF_sh_old = LF_sh
    LF_sh = np.multiply(np.divide(LF_sh, max_v), clip_max)

    # find scale constant
    tmp = LF_sh_old
    tmp[LF_sh_old == 0] = 1
    alpha = np.divide(LF_sh, tmp)
    alpha[LF_sh_old == 0] = 1
    alpha[np.isnan(alpha)] = 1
    alpha[np.isinf(alpha)] = 1
    del LF_sh_old
  else:
    alpha = 1

  # glossy LF
  LF_specular = np.multiply(LF_gc, np.add(LF_gd, LF_gi))
  LF_specular = np.multiply(alpha, LF_specular)
  # input LF
  LF = np.add(np.multiply(LF_dc, LF_sh),LF_specular)

  dset_LF_albedo = file.create_dataset('LF_albedo', data=LF_albedo)
  dset_LF_sh = file.create_dataset('LF_sh', data=LF_sh)
  dset_LF_specular = file.create_dataset('LF_specular', data=LF_specular)

  lf_tools.save_image(data_source + '\\' + 'input_' + lf_name, lf_tools.cv(LF))
  lf_tools.save_image(data_source + '\\' + 'albedo_' + lf_name, lf_tools.cv(LF_albedo))
  lf_tools.save_image(data_source + '\\' + 'sh_' + lf_name + '_' + np.str(np.amax(LF_sh)), lf_tools.cv(LF_sh))
  lf_tools.save_image(data_source + '\\' + 'specular_' + lf_name + '_' + np.str(np.amax(LF_specular)),
                      lf_tools.cv(LF_specular))

  # input LF

  dset_LF = file.create_dataset('LF', data = LF)

  disp = file_io.read_disparity( data_folder )
  disp_gt = np.array( disp[0] )
  disp_gt = np.flip( disp_gt,0 )
  dset_disp = file.create_dataset('LF_disp', data=disp_gt)

  # next dataset
  print(' done.')
    
