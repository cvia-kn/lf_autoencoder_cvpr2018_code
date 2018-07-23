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

# patch size.
px = 96 # 48
py = 96 # 48

# number of views in H/V/ direction
# input data must match this.
nviews = 9
clip_max = 2

# block step size.

sx = 32# 16
sy = 32# 16

# output file to write to
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!! careful: overwrite mode !!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
# previous training data will be erased.

training_data_dir = "H:\\trainData\\"
training_data_filename = 'lf_patch_autoencoder_light_300_96.hdf5'
file = h5py.File( training_data_dir + training_data_filename, 'w' )


#
#data_folders = ( ( "training", "boxes" ), )
# data_folders = data_folders_base + data_folders_add
data_source = "H:\\CNN_data_light_190218\\train\\300"
data_folders = os.listdir(data_source)



# EPI patches, nviews x patch size x patch size x channels
# horizontal and vertical direction (to get crosshair)
dset_v = file.create_dataset( 'stacks_v', ( nviews, py,px, 3, 1 ),
                              chunks = ( nviews, py,px, 3, 1 ),
                              maxshape = ( nviews, py,px, 3, None ) )

dset_h = file.create_dataset( 'stacks_h', ( nviews, py,px, 3, 1 ),
                              chunks = ( nviews, py,px, 3, 1 ),
                              maxshape = ( nviews, py,px, 3, None ) )

# dataset for corresponding depth patches
dset_depth = file.create_dataset( 'depth', ( py,px, 1 ),
                                  chunks = ( py,px, 1 ),
                                  maxshape = ( py,px, None ) )

# dataset for corresponding diffuse patches
dset_diffuse_v = file.create_dataset( 'diffuse_v', ( nviews, py,px, 3, 1 ),
                              chunks = ( nviews, py,px, 3, 1 ),
                              maxshape = ( nviews, py,px, 3, None ) )
dset_diffuse_h = file.create_dataset( 'diffuse_h', ( nviews, py,px, 3, 1 ),
                              chunks = ( nviews, py,px, 3, 1 ),
                              maxshape = ( nviews, py,px, 3, None ) )

# dataset for corresponding specular patches
dset_specular_v = file.create_dataset( 'specular_v', ( nviews, py,px, 3, 1 ),
                              chunks = ( nviews, py,px, 3, 1 ),
                              maxshape = ( nviews, py,px, 3, None ) )
dset_specular_h = file.create_dataset( 'specular_h', ( nviews, py,px, 3, 1 ),
                              chunks = ( nviews, py,px, 3, 1 ),
                              maxshape = ( nviews, py,px, 3, None ) )

# dataset for correcponsing center view patch (to train joint upsampling)
# ideally, would want to reconstruct full 4D LF patch, but probably too memory-intensive
# keep for future work
dset_cv = file.create_dataset( 'cv', ( py,px, 3, 1 ),
                               chunks = ( py,px, 3, 1 ),
                               maxshape = ( py,px, 3, None ) )

dset_diffuse = file.create_dataset( 'diffuse', ( py,px, 3, 1 ),
                               chunks = ( py,px, 3, 1 ),
                               maxshape = ( py,px, 3, None ) )

dset_specular = file.create_dataset( 'specular', ( py,px, 3, 1 ),
                               chunks = ( py,px, 3, 1 ),
                               maxshape = ( py,px, 3, None ) )


#
# loop over all datasets, write out each dataset in patches
# to feed to autoencoder in random order
#
index = 0
for lf_name in data_folders:

  # data_folder = "/data/lfa/" + lf_name[0] + "/" + lf_name[1] + "/"
  data_folder = os.path.join(data_source,lf_name)
  # read diffuse color
  LF_dc = file_io.read_lightfield_intrinsic_crosshair(data_folder, 'dc')
  # read diffuse direct
  LF_dd = file_io.read_lightfield_intrinsic_crosshair(data_folder, 'dd')
  # read diffuse indirect
  LF_di = file_io.read_lightfield_intrinsic_crosshair(data_folder, 'di')
  # read glossy color
  LF_gc = file_io.read_lightfield_intrinsic_crosshair(data_folder, 'gc')
  # read glossy direct
  LF_gd = file_io.read_lightfield_intrinsic_crosshair(data_folder, 'gd')
  # read glossy indirect
  LF_gi = file_io.read_lightfield_intrinsic_crosshair(data_folder, 'gi')


  # albedo LF
  LF_albedo = LF_dc
  # shading LF
  LF_sh = np.add(LF_dd, LF_di)

  min_v = np.amin(LF_sh)
  max_v = np.amax(LF_sh)
  if max_v > 2:
      print('rescaling')
      LF_sh_old = LF_sh
      LF_sh  = np.multiply(np.divide(LF_sh,max_v), clip_max)

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
  # diffuse LF
  LF_diffuse = np.multiply(LF_albedo,LF_sh)
  # show center view
  cv_diffuse = lf_tools.cv(LF_diffuse)
  # show center view
  cv_specular = lf_tools.cv(LF_specular)
  # lf_tools.save_image( training_data_dir + 'specular' +lf_name, cv_specular)
  # input LF
  LF = np.add(LF_diffuse, LF_specular)
  cv_gt = lf_tools.cv(LF)

  # imean = 0.3
  # factor = imean / np.mean(cv_gt)
  # LF_diffuse = LF_diffuse*factor
  # LF_specular = LF_specular*factor
  # LF = np.add(LF_diffuse, LF_specular)
  # cv_gt = lf_tools.cv(LF)

  disp = file_io.read_disparity( data_folder )
  disp_gt = np.array( disp[0] )
  disp_gt = np.flip( disp_gt,0 )

  lf_tools.save_image(training_data_dir + 'input' + lf_name, cv_gt)
  # lf_tools.save_image(training_data_dir + 'input' + lf_name, cv_gt)
  
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
      (stack_v, stack_h) = lf_tools.epi_stacks( LF, y, x, py, px )
      # make sure the direction of the view shift is the first spatial dimension
      stack_h = np.transpose( stack_h, (0, 2, 1, 3) )

      (diffuse_stack_v, diffuse_stack_h) = lf_tools.epi_stacks(LF_diffuse, y, x, py, px)
      diffuse_stack_h = np.transpose(diffuse_stack_h, (0, 2, 1, 3))

      (specular_stack_v, specular_stack_h) = lf_tools.epi_stacks(LF_specular, y, x, py, px)
      specular_stack_h = np.transpose(specular_stack_h, (0, 2, 1, 3))
      
      depth = disp_gt[ y:y+py, x:x+px ]
      cv = cv_gt[ y:y+py, x:x+px ]
      diffuse = cv_diffuse[y:y + py, x:x + px]
      specular = cv_specular[y:y + py, x:x + px]

      # write to respective HDF5 datasets
      dset_v.resize( index+1, 4 )
      dset_v[ :,:,:,:, index ] = stack_v

      dset_h.resize( index+1, 4 )
      dset_h[ :,:,:,:, index ] = stack_h

      dset_diffuse_v.resize(index + 1, 4)
      dset_diffuse_v[:, :, :, :, index] = diffuse_stack_v

      dset_diffuse_h.resize(index + 1, 4)
      dset_diffuse_h[:, :, :, :, index] = diffuse_stack_h

      dset_specular_v.resize(index + 1, 4)
      dset_specular_v[:, :, :, :, index] = specular_stack_v

      dset_specular_h.resize(index + 1, 4)
      dset_specular_h[:, :, :, :, index] = specular_stack_h

      dset_depth.resize( index+1, 2 )
      dset_depth[ :,:, index ] = depth

      dset_cv.resize( index+1, 3 )
      dset_cv[ :,:,:, index ] = cv

      dset_diffuse.resize( index+1, 3 )
      dset_diffuse[ :,:,:, index ] = diffuse

      dset_specular.resize( index+1, 3 )
      dset_specular[ :,:,:, index ] = specular
      
      # next patch
      index = index + 1

  # next dataset
  print(' done.')
    
