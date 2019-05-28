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
import lf_tools



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

training_data_dir      = "H:\\trainData\\"
training_data_filename = 'lf_patch_intrinsic_lytro_flowers.hdf5'
file = h5py.File( training_data_dir + training_data_filename, 'w' )


#
#data_folders = ( ( "training", "boxes" ), )
# data_folders = data_folders_base + data_folders_add
data_source = "E:\\LYTRODATA\\flowers"
data_folders = os.listdir(data_source)

# EPI patches, nviews x patch size x patch size x channels
# horizontal and vertical direction (to get crosshair)
dset_v = file.create_dataset( 'stacks_v', ( nviews, py,px, 3, 1 ),
                              chunks = ( nviews, py,px, 3, 1 ),
                              maxshape = ( nviews, py,px, 3, None ) )

dset_h = file.create_dataset( 'stacks_h', ( nviews, py,px, 3, 1 ),
                              chunks = ( nviews, py,px, 3, 1 ),
                              maxshape = ( nviews, py,px, 3, None ) )

# dataset for correcponsing center view patch (to train joint upsampling)
# ideally, would want to reconstruct full 4D LF patch, but probably too memory-intensive
# keep for future work
dset_cv = file.create_dataset( 'cv', ( py,px, 3, 1 ),
                               chunks = ( py,px, 3, 1 ),
                               maxshape = ( py,px, 3, None ) )

#
# loop over all datasets, write out each dataset in patches
# to feed to autoencoder in random order
#
index = 0
for lf_name in data_folders:

  data_file = os.path.join(data_source,lf_name)

  # input LF
  mat_content = h5py.File(data_file, 'r')
  LF = mat_content['LF'].value
  LF = np.transpose(LF, (4, 3, 2, 1, 0))

  cv_gt = lf_tools.cv( LF )
  lf_tools.save_image(training_data_dir + 'input' + lf_name, cv_gt)

  # write out one individual light field
  # block count
  cx = np.int32( ( LF.shape[3] - px) / sx ) + 1  
  cy = np.int32( ( LF.shape[2] - py) / sy ) + 1

  for by in np.arange( 0, cy ):
    sys.stdout.write( '.' )
    sys.stdout.flush()
    
    for bx in np.arange( 0, cx ):

      x = bx * sx
      y = by * sx

      # extract data
      (stack_v, stack_h) = lf_tools.epi_stacks( LF, y, x, py, px )
      # make sure the direction of the view shift is the first spatial dimension
      stack_h = np.transpose( stack_h, (0, 2, 1, 3) )

      cv = cv_gt[ y:y+py, x:x+px ]

      #code.interact( local=locals() )

      
      # write to respective HDF5 datasets
      dset_v.resize( index+1, 4 )
      dset_v[ :,:,:,:, index ] = stack_v

      dset_h.resize( index+1, 4 )
      dset_h[ :,:,:,:, index ] = stack_h

      dset_cv.resize( index+1, 3 )
      dset_cv[ :,:,:, index ] = cv

      # next patch
      index = index + 1

  # next dataset
  print(' done.')
    
