#
# A bunch of useful helper functions to work with
# the light field data.
#
# (c) Bastian Goldluecke, Uni Konstanz
# bastian.goldluecke@uni.kn
# License: Creative Commons CC BY-SA 4.0
#

import numpy as np
import matplotlib.pyplot as plt



# returns two epipolar plane image stacks (horizontal/vertical),
# block size (xs,ys), block location (x,y), both in pixels.
def epi_stacks( LF, y, x, ys, xs ):

  T = np.int32( LF.shape[0] )
  cv_v = np.int32( (T - 1) / 2 )
  S = np.int32( LF.shape[1] )
  cv_h = np.int32( (S - 1) / 2 )
  stack_h = LF[ cv_v, :, y:y+ys, x:x+xs, : ]
  stack_v = LF[ :, cv_h, y:y+ys, x:x+xs, : ]
  return (stack_h, stack_v)

# returns center view
def cv( LF ):

  T = np.int32( LF.shape[0] )
  cv_v = np.int32( (T - 1) / 2 )
  S = np.int32( LF.shape[1] )
  cv_h = np.int32( (S - 1) / 2 )
  return LF[ cv_v, cv_h, :,:,: ]


# show an image (with a bunch of checks)
def show( img, cmap='gray' ):

  if len( img.shape ) == 2:
    img = img[ :,:,np.newaxis ]
    
  if img.shape[2]==1:
    img = img[ :,:,0]
    #img = np.clip( img, 0.0, 1.0 )
    imgplot = plt.imshow( img, interpolation='none', cmap=cmap )
  else:
    imgplot = plt.imshow( img, interpolation='none' )
  plt.show( block=False )


def augment_data(input, idx):
  size = input['stacks_v'][idx].shape
  a = np.tile(np.random.randn(1,1,1,3) / 8 + 1, (size[0], size[1], size[2],1))
  b = np.tile(np.random.randn(1,1,1,3) / 8, (size[0], size[1], size[2],1))

  stacks_v = input['stacks_v'][idx]
  stacks_h = input['stacks_h'][idx]
  stacks_v = augment_albedo(a, b, stacks_v)
  stacks_h = augment_albedo(a, b, stacks_h)

  input['stacks_v'][idx] = stacks_v
  input['stacks_h'][idx] = stacks_h

  return(input)

def augment_albedo(a,b,albedo):
  out = np.multiply( a, albedo) + b
  out = out - np.minimum(0, np.amin(out))
  out = np.divide(out, np.maximum(np.amax(out), 1))
  return(out)

def augment_data_intrinsic(input, idx):
  size = input['stacks_v'][idx].shape

  d = np.tile(np.abs(np.random.randn(1, 1, 1,3) / 8 + 1), (size[0], size[1], size[2],1))
  c = np.abs(np.random.randn(1) / 4 + 1)
  diffuse_v = input['diffuse_v'][idx]
  diffuse_h = input['diffuse_h'][idx]

  input['diffuse_v'][idx] = np.multiply(diffuse_v,d)
  input['diffuse_h'][idx] = np.multiply(diffuse_h,d)

  specular_v = input['specular_v'][idx]
  specular_h = input['specular_h'][idx]

  specular_v = np.multiply(c, specular_v)
  specular_h = np.multiply(c, specular_h)

  input['specular_v'][idx] = specular_v
  input['specular_h'][idx] = specular_h

  input['stacks_v'][idx] = input['diffuse_v'][idx] + input['specular_v'][idx]
  input['stacks_h'][idx] = input['diffuse_h'][idx] + input['specular_h'][idx]

  return(input)



# visualize an element of a batch for training/test
def show_batch( batch, n ):
  ctr = 4

  # vertical stack
  plt.subplot(2, 2, 1)
  plt.imshow( batch[ 'stacks_v' ][ n, :,:, 24,: ] )
  
  # horizontal stack
  plt.subplot(2, 2, 2)
  plt.imshow( batch[ 'stacks_h' ][ n, :,:, 24,: ] )

  # vertical stack center
  plt.subplot(2, 2, 3)
  plt.imshow( batch[ 'stacks_v' ][ n, ctr, :,:,: ] )

  # horizontal stack center
  plt.subplot(2, 2, 4)
  plt.imshow( batch[ 'stacks_h' ][ n, ctr, :,:,: ] )  
    
  plt.show()
