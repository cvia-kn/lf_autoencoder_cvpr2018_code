#
# A bunch of useful helper functions to work with
# the light field data.
#
# (c) Bastian Goldluecke, Uni Konstanz
# bastian.goldluecke@uni.kn
# License: Creative Commons CC BY-SA 4.0
#

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.misc as sc
import numpy.matlib
from skimage import feature
from scipy import ndimage

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

def epi_disp( LF, y, x, ys, xs ):

  T = np.int32( LF.shape[0] )
  cv_v = np.int32( (T - 1) / 2 )
  S = np.int32( LF.shape[1] )
  cv_h = np.int32( (S - 1) / 2 )
  stack_h = LF[ cv_v, :, y:y+ys, x:x+xs]
  stack_v = LF[ :, cv_h, y:y+ys, x:x+xs]
  return (stack_v, stack_h)

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

  #code.interact(local=locals())
  plt.show( block=False )

def save_image(im_path, im):
  sc.imsave(im_path + '.png', im)

# show a slicing through an array along given dimension indices
def show_slice( array, dims ):

  ext = [ slice(None) ] * len(array)
  for i in range(0,len(dims)):
    ext[ dims[i] ] = slice( 0,-1 )

  np.allclose( array[ext], subset )
  show( subset )


def plane_data(data, H_in, W_in, S_in, T_in):
    batch_size = data.shape[2]
    data_out = np.zeros((batch_size, H_in*S_in, W_in*T_in, 3),dtype = np.float32 )
    for i in range(0,S_in):
        for j in range(0,T_in):
            yrange = range(i, H_in * S_in, S_in)
            xrange = range(j, W_in * T_in, T_in)
            data_out[:,np.array(np.matlib.repmat(yrange,len(xrange),1).transpose().flatten())
                                     , np.array(np.matlib.repmat(xrange,1,len(yrange))[0]),:] = np.reshape(data[i,j,:,:,:,:],(batch_size, H_in*W_in,3))

    return data_out
# size [batch_size, H*s, T*W, 3]

def canny_edge(disparity_lf):
    strel = np.ones([5,5])
    strel[0,0] = 0
    strel[0,1] = 0
    strel[0,3] = 0
    strel[0,4] = 0
    strel[1,0] = 0
    strel[3,0] = 0
    strel[4,0] = 0
    tst1 = np.flip(strel, axis=1)
    tst0 = np.flip(strel, axis=0)
    tst = np.multiply(np.multiply(strel, tst1), tst0)
    strel = tst
    edge_lf = np.zeros(disparity_lf.shape)
    for s in range(0,9):
        for t in range(0,9):
            if s == 4 or t == 4:
                tmp = feature.canny(disparity_lf[s,t,:,:])
                edge_lf[s, t, :, :] = ndimage.morphology.binary_dilation(tmp, structure = strel)

    return edge_lf