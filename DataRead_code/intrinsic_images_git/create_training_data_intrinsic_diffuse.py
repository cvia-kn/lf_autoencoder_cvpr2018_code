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



# python tools for our lf database
import file_io
# additional light field tools
import lf_tools



# OUTPUT CONFIGURATION

# patch size. patches of this size will be extracted and stored
# must remain fixed, hard-coded in NN
px = 96 #48
py = 96 #48

# number of views in H/V/ direction
# input data must match this.
nviews = 9

# block step size. this is only 16, as we keep only the center 16x16 block
# of each decoded patch (reason: reconstruction quality will probably strongly
# degrade towards the boundaries).
# 
# TODO: test whether the block step can be decreased during decoding for speedup.
#
sx = 32 #16
sy = 32 #16

# output file to write to
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!! careful: overwrite mode !!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
# previous training data will be erased.

training_data_dir      = "D:\\Python\\DataRead\\trainData\\"
training_data_filename = 'lf_patch_intrinsic3_diffuse_96.hdf5'
file = h5py.File( training_data_dir + training_data_filename, 'w' )


#
#data_folders = ( ( "training", "boxes" ), )
# data_folders = data_folders_base + data_folders_add
data_source = "E:\\CNN_data\\diffuse\\3"
data_folders = os.listdir(data_source)

dset_v = file.create_dataset( 'stacks_v', ( 9, py,px, 3, 1 ),
                              chunks = ( 9, py,px, 3, 1 ),
                              maxshape = ( 9, py,px, 3, None ) )

dset_h = file.create_dataset( 'stacks_h', ( 9, py,px, 3, 1 ),
                              chunks = ( 9, py,px, 3, 1 ),
                              maxshape = ( 9, py,px, 3, None ) )

# dataset for corresponding depth patches
dset_depth = file.create_dataset( 'depth', ( py,px, 1 ),
                                  chunks = ( py,px, 1 ),
                                  maxshape = ( py,px, None ) )

# dataset for corresponding albedo patches
dset_albedo_v = file.create_dataset( 'albedo_v', ( 9, py,px, 3, 1 ),
                              chunks = ( 9, py,px, 3, 1 ),
                              maxshape = ( 9, py,px, 3, None ) )
dset_albedo_h = file.create_dataset( 'albedo_h', ( 9, py,px, 3, 1 ),
                              chunks = ( 9, py,px, 3, 1 ),
                              maxshape = ( 9, py,px, 3, None ) )

# dataset for corresponding shading patches
dset_sh_v = file.create_dataset( 'sh_v', ( 9, py,px, 3, 1 ),
                              chunks = ( 9, py,px, 3, 1 ),
                              maxshape = ( 9, py,px, 3, None ) )
dset_sh_h = file.create_dataset( 'sh_h', ( 9, py,px, 3, 1 ),
                              chunks = ( 9, py,px, 3, 1 ),
                              maxshape = ( 9, py,px, 3, None ) )

# dataset for corresponding specular patches
dset_specular_v = file.create_dataset( 'specular_v', ( 9, py,px, 3, 1 ),
                              chunks = ( 9, py,px, 3, 1 ),
                              maxshape = ( 9, py,px, 3, None ) )
dset_specular_h = file.create_dataset( 'specular_h', ( 9, py,px, 3, 1 ),
                              chunks = ( 9, py,px, 3, 1 ),
                              maxshape = ( 9, py,px, 3, None ) )
#
# loop over all datasets, write out each dataset in patches
# to feed to autoencoder in random order
#

# data_folders = {data_folders[1],}
index = 0
for lf_name in data_folders:

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

  # albedo LF
  LF_albedo = LF_dc
  # shading LF
  LF_sh = np.add(LF_dd, LF_di)
  # glossy LF
  LF_specular = np.multiply(LF_gc, np.add(LF_gd, LF_gi))*0
  # input LF
  LF = np.add(np.multiply(LF_dc, LF_sh),LF_specular)

  disp = file_io.read_disparity( data_folder )
  disp_gt = np.array( disp[0] )
  disp_gt = np.flip( disp_gt,0 )
  
  # maybe we need those, probably not.
  param_dict = file_io.read_parameters(data_folder)

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
      (stack_v, stack_h) = lf_tools.epi_stacks(LF, y, x, py, px)
      # make sure the direction of the view shift is the first spatial dimension
      stack_h = np.transpose(stack_h, (0, 2, 1, 3))

      (stack_v_albedo, stack_h_albedo) = lf_tools.epi_stacks( LF_albedo, y, x, py, px )
      # make sure the direction of the view shift is the first spatial dimension
      stack_h_albedo = np.transpose( stack_h_albedo, (0, 2, 1, 3) )

      (stack_v_sh, stack_h_sh) = lf_tools.epi_stacks( LF_sh, y, x, py, px )
      # make sure the direction of the view shift is the first spatial dimension
      stack_h_sh = np.transpose( stack_h_sh, (0, 2, 1, 3) )

      (stack_v_specular, stack_h_specular) = lf_tools.epi_stacks(LF_specular, y, x, py, px)
      stack_h_specular = np.transpose(stack_h_specular, (0, 2, 1, 3))
      
      depth = disp_gt[ y:y+py, x:x+px ]

      # write to respective HDF5 datasets
      # input
      dset_v.resize(index + 1, 4)
      dset_v[:, :, :, :, index] = stack_v

      dset_h.resize(index + 1, 4)
      dset_h[:, :, :, :, index] = stack_h

      # albedo
      dset_albedo_v.resize(index + 1, 4)
      dset_albedo_v[:, :, :, :, index] = stack_v_albedo

      dset_albedo_h.resize(index + 1, 4)
      dset_albedo_h[:, :, :, :, index] = stack_h_albedo

      # shading

      dset_sh_v.resize(index + 1, 4)
      dset_sh_v[:, :, :, :, index] = stack_v_sh

      dset_sh_h.resize(index + 1, 4)
      dset_sh_h[:, :, :, :, index] = stack_h_sh

      # specularity

      dset_specular_v.resize(index + 1, 4)
      dset_specular_v[:, :, :, :, index] = stack_v_specular

      dset_specular_h.resize(index + 1, 4)
      dset_specular_h[:, :, :, :, index] = stack_h_specular

      dset_depth.resize( index+1, 2 )
      dset_depth[ :,:, index ] = depth
      
      # next patch
      index = index + 1

  # next dataset
  print(' done.')
    
