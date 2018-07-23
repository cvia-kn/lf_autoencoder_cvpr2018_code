# define the configuration (hyperparameters) for the data format
# this is hard-coded now in all the training data, should not be changed anymore


import lf_tools
import numpy as np

# general config params
data_config = {
    # patch size
    'D' : 9,
    'H' : 48,
    'W' : 48,
    # patch stepping
    'SX' : 16,
    'SY' : 16,
    # depth range and number of labels
    'dmin' : -3.5,
    'dmax' :  3.5,
}



# get patch at specified block coordinates
def get_patch( LF, cv, disp, by, bx ):

  patch = dict()

  # compute actual coordinates
  y = by * data_config['SY']
  x = bx * data_config['SX']
  py = data_config['H']
  px = data_config['W']
  
  # extract data
  # (stack_h, stack_v) = lf_tools.epi_stacks( LF, y, x, py, px )
  (stack_v, stack_h) = lf_tools.epi_stacks( LF, y, x, py, px )

  # make sure the direction of the view shift is the first spatial dimension
  stack_h = np.transpose( stack_h, (0, 2,1, 3) )
  patch[ 'stack_v' ] = stack_v
  patch[ 'stack_h' ] = stack_h
  patch[ 'cv' ]      = cv[ y:y+py, x:x+px ]

  if disp is not None:
    depth = disp[ y:y+py, x:x+px ]
    depth = np.minimum( data_config['dmax'], depth )
    depth = np.maximum( data_config['dmin'], depth )

    patch[ 'depth' ] = depth


  return patch
