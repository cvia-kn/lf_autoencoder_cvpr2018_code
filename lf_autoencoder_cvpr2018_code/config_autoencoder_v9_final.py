# define the configuration (hyperparameters) for the residual autoencoder
# for this type of network.


# NETWORK MODEL NAME
network_model = 'current_v9'

# CURRENT TRAINING DATASET
training_data = [
    '/data/cnns/training_data/lf_patch_autoencoder1.hdf5',
]

# NETWORK LAYOUT HYPERPARAMETERS

# general config params
config = {
    # flag whether we want to train for RGB (might require more
    # changes in other files, can't remember right now)
    'rgb'                  : True,

    # maximum layer which will be initialized (deprecated)
    'max_layer'            : 100,

    # this will log every tensor being allocated,
    # very spammy, but useful for debugging
    'log_device_placement' : False,
}


# encoder for 48 x 48 patch, 9 views, RGB
D = 9
H = 48
W = 48
if config['rgb']:
    C = 3
else:
    C = 1

# Number of features in the layers
L = 16
L0 = 24
L1 = 2*L
L2 = 4*L
L3 = 6*L
L4 = 8*L
L5 = 10*L
L6 = 12*L


# Encoder stack for downwards convolution
layers = dict()

layers[ 'autoencoder' ] = [
    { 'conv'   : [ 3,3,3, C, L0 ],
      'stride' : [ 1,1,1, 1, 1 ]
    },
    { 'conv'   : [ 3,3,3, L0, L0 ],
      'stride' : [ 1,1,1, 1, 1 ]
    },
    { 'conv'   : [ 3,3,3, L0, L1 ],
      'stride' : [ 1,1, 2,2, 1 ]
    },
    # resolution now 9 x 24 x 24
    { 'conv'   : [ 3,3,3, L1, L1 ],
      'stride' : [ 1,1,1, 1, 1 ]
    },
    { 'conv'   : [ 3,3,3, L1, L1 ],
      'stride' : [ 1,1,1, 1, 1 ]
    },
    { 'conv'   : [ 3,3,3, L1, L2 ],
      'stride' : [ 1,2, 1,1, 1 ]
    },
    # resolution now 5 x 24 x 24
    { 'conv'   : [ 3,3,3, L2, L2 ],
      'stride' : [ 1,1,1, 1, 1 ]
    },
    { 'conv'   : [ 3,3,3, L2, L2 ],
      'stride' : [ 1,1,1, 1, 1 ]
    },
    { 'conv'   : [ 3,3,3, L2, L3 ],
      'stride' : [ 1,1, 2,2, 1 ]
    },
    # resolution now 5 x 12 x 12
    { 'conv'   : [ 3,3,3, L3, L3 ],
      'stride' : [ 1,1,1, 1, 1 ]
    },
    { 'conv'   : [ 3,3,3, L3, L3 ],
      'stride' : [ 1,1,1, 1, 1 ]
    },
    { 'conv'   : [ 3,3,3, L3, L4 ],
      'stride' : [ 1,1, 2,2, 1 ]
    },
    # resolution now 5 x 6 x 6
    { 'conv'   : [ 3,3,3, L4, L4 ],
      'stride' : [ 1,1,1, 1, 1 ]
    },
    { 'conv'   : [ 3,3,3, L4, L4 ],
      'stride' : [ 1,1,1, 1, 1 ]
    },
    { 'conv'   : [ 3,3,3, L4, L5 ],
      'stride' : [ 1,2, 1,1, 1 ]
    },
    # resolution now 3 x 6 x 6
    { 'conv'   : [ 3,3,3, L5, L5 ],
      'stride' : [ 1,1,1, 1, 1 ]
    },
    { 'conv'   : [ 3,3,3, L5, L5 ],
      'stride' : [ 1,1,1, 1, 1 ]
    },
    { 'conv'   : [ 3,3,3, L5, L6 ],
      'stride' : [ 1,1, 2,2, 1 ]
    },
    # resolution now 3 x 3 x 3
]


# chain of dense layers to form small bottleneck (can be empty)
layers[ 'autoencoder_nodes' ] = []
layers[ '2D_decoder_nodes' ] = []
layers[ 'preferred_gpu' ] = 0
# if skip-connections are used
pinhole_connections = True
# layers where features should be concatenated. Attention: always ends with the last layer
# For example here we concatenate features on the 17th layers for the linked decoders: diffuse and specular
connect_layers = [ 17,18]

#
# 3D DECODERS
#
# Generates one default pathway for the EPI stacks
# In the training data, there must be corresponding
# data streams 'decoder_name_v' and 'decoder_name_h',
# which are used for loss computation.
#
decoders_3D = [
    { 'id':      'stacks',
      'channels': C,
      'preferred_gpu' : 1,
      'loss_fn':  'L2',
      'train':    True,
      'weight':   1.0,
    },
]

linked_decoders_3D = [
    { 'id':      'diffuse',
      'channels': C,
      'preferred_gpu' : 2,
      'loss_fn':  'L2',
      'train':    True,
      'weight':   1.0,
    },
    { 'id':      'specular',
      'channels': C,
      'preferred_gpu' : 2,
      'loss_fn':  'L2',
      'train':    True,
      'weight':   1.0,
    },
]



#
# 2D DECODERS
#
# Each one generates a 2D upsampling pathway next to the
# two normal autoencoder pipes.

decoders_2D = [
    { 'id': 'depth',
      'channels': 1,
      'preferred_gpu' : 3,
      'loss_fn':  'L2',
      'train':    True,
      'weight':   1.0,
    },
]

# MINIMIZERS
minimizers = [
  # autoencoder
  { 'id' : 'AE',
    'losses_3D' : [ 'stacks', 'diffuse_specular_sum' ],
    'optimizer' : 'Adam',
    'preferred_gpu' : 1,
    'step_size' : 1e-4,
  },
  # diffuse and specular
  { 'id'        : 'DS',
    'losses_3D' : [ 'diffuse', 'specular' ],
    'optimizer' : 'Adam',
    'preferred_gpu' : 2,
    'step_size' : 1e-4,
  },
  # disparity
  { 'id'        : 'D',
    'losses_2D' : [ 'depth' ],
    'optimizer' : 'Adam',
    'preferred_gpu' : 3,
    'step_size' : 1e-4,
  },
]


# TRAINING HYPERPARAMETERS
training = dict()

# subsets to split training data into
# by default, only 'training' will be used for training, but the results
# on mini-batches on 'validation' will also be logged to check model performance.
# note, split will be performed based upon a random shuffle with filename hash
# as seed, thus, should be always the same for the same file.
#
training[ 'subsets' ] = {
  'validation'   : 0.05,
  'training'     : 0.95,
}


# number of samples per mini-batch
training[ 'samples_per_batch' ] = 10

# log interval (every # mini-batches per dataset)
training[ 'log_interval' ] = 5

# save interval (every # iterations over all datasets)
training[ 'save_interval' ] = 15

# noise to be added on each input patch
# (NOT on the decoding result)
training[ 'noise_sigma' ] = 0.0

# decay parameter for batch normalization
# should be larger for larger datasets
training[ 'batch_norm_decay' ]  = 0.9
# flag whether BN center param should be used
training[ 'batch_norm_center' ] = False
# flag whether BN scale param should be used
training[ 'batch_norm_scale' ]  = False
# flag whether BN should be zero debiased (extra param)
training[ 'batch_norm_zero_debias' ]  = False

eval_res = {
    'h_mask': 40,
    'w_mask': 40,
    'm': 4,
    'min_mask': 0.1,
    'result_folder': "./results/",
    'test_data_folder': "./test_data/"
}