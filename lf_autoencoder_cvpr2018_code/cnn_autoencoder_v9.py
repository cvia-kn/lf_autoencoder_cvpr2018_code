# Class definition for the combined CRY network
# drops: deep regression on angular patch stacks
#
# in this version, we take great care to have nice
# variable scope names.
#
# start session
import code
import tensorflow as tf
import numpy as np

import libs.layers as layers

loss_min_coord_3D = 0
loss_max_coord_3D = 48

loss_min_coord_2D = 0
loss_max_coord_2D = 48

# main class defined in module
class create_cnn:

  def __init__( self, config ):

    # config (hyperparameters)
    self.config = config
    self.max_layer = config.config[ 'max_layer' ]

    # we get two input paths for autoencoding:
    # 1. vertical epi stack in stack_v
    # 2. horizontal epi stack in stack_h

    # both stacks have 9 views, patch size 16x16 + 16 overlap on all sides,
    # for a total of 48x48.
    if config.config['rgb']:
      self.C = 3
    else:
      self.C = 1

    self.D = config.D
    self.H = config.H
    self.W = config.W

    # regularization weights
    self.beta = 0.0001

    # input layers
    with tf.device( '/device:GPU:%i' % ( self.config.layers['preferred_gpu'] ) ):
      with tf.variable_scope( 'input' ):

        self.stack_v = tf.placeholder(tf.float32, shape=[None, self.D, self.H, self.W, self.C] )
        self.stack_h = tf.placeholder(tf.float32, shape=[None, self.D, self.H, self.W, self.C] )
        self.stack_shape = self.stack_v.shape.as_list()
        self.stack_shape[ 0 ] = -1

        self.phase = tf.placeholder(tf.bool, name='phase')
        self.keep_prob = tf.placeholder(tf.float32)
        self.noise_sigma = tf.placeholder(tf.float32)

    # FEATURE LAYERS
    self.encoder_variables = []
    self.conv_layers_v = []
    self.conv_layers_h = []
    self.upconv_layers_v = []
    self.upconv_layers_h = []
    self.features_v = None
    self.features_h = None
    self.decoders_3D = dict()
    self.decoders_2D = dict()
    self.minimizers = dict()

    self.batch_size = tf.shape(self.stack_v)[0]

    self.create_variables()
    self.create_encoder_layers()
    self.create_dense_feature_layers()

    self.create_3D_decoders()
    self.create_2D_decoders()

    self.setup_losses()

  #
  # CREATE NETWORK VARIABLES
  #
  # Done globally, as the autoencoder/-decoder variables are shared by
  # horizontal/vertical EPI upsampling

  def create_variables( self ):

    print( '  initializing model autoencoder variables' )

    with tf.device( '/device:GPU:%i' % ( self.config.layers['preferred_gpu'] ) ):
      layout = self.config.layers['autoencoder']
      first_layer = len( self.encoder_variables )
      last_layer = min( len(layout), self.max_layer )

      for i in range( first_layer, last_layer ):

        layer_id = "encoder_%i" % i
        print( '    creating encoder variables ' + layer_id )

        # generate variables (if necessary)
        self.encoder_variables.append( layers.encoder_variables( layer_id, layout[i] ) )

  #
  # CREATE CONVOLUTIONAL ENCODING PATH
  #
  def create_encoder_layers( self ):

    print( '  initializing model layer structure' )
    layout = self.config.layers['autoencoder']

    # followup layers have same kernel sizes
    with tf.device( '/device:GPU:%i' % ( self.config.layers['preferred_gpu'] ) ):
      if self.features_v == None:
        self.features_v = self.stack_v
        self.features_v = tf.reshape( self.features_v, self.stack_shape )
      if self.features_h == None:
        self.features_h = self.stack_h
        self.features_h = tf.reshape( self.features_h, self.stack_shape )

      first_layer = len( self.conv_layers_v )
      last_layer = min( len(layout), self.max_layer )
      for i in range( first_layer, last_layer ):

        layer_id_v = "v_ae_%i" % i
        layer_id_h = "h_ae_%i" % i
        print( '    generating downconvolution layer structure ' + layer_id_v )

        # setup encoder layer and connect to previous one
        self.conv_layers_v.append( layers.layer_conv3d( layer_id_v,
                                                        self.encoder_variables[i],
                                                        self.features_v,
                                                        self.phase,
                                                        self.config.training ))

        # setup encoder layer and connect to previous one
        self.conv_layers_h.append( layers.layer_conv3d( layer_id_h,
                                                        self.encoder_variables[i],
                                                        self.features_h,
                                                        self.phase,
                                                        self.config.training ))

        # update layer shapes
        self.encoder_variables[i].input_shape = self.conv_layers_v[i].input_shape
        self.encoder_variables[i].output_shape = self.conv_layers_v[i].output_shape

        # final encoder layer: vertical/horizontal features
        self.features_v = self.conv_layers_v[-1].out
        self.features_h = self.conv_layers_h[-1].out

  #
  # CREATE DENSE FEATURE LAYERS
  #
  # this fuses horizontal/vertical path at the bottom level
  # optionally, no extra feature layers (then final nodes are just flattened)
  #
  def create_dense_feature_layers( self ):

    print( '  creating dense layers' )
    with tf.device( '/device:GPU:%i' % ( self.config.layers['preferred_gpu'] ) ):
      self.feature_shape = self.features_v.shape.as_list()
      sh = self.feature_shape
      self.encoder_input_size = sh[1] * sh[2] * sh[3] * sh[4]

      # setup shared feature space between horizontal/vertical encoder
      self.features = tf.concat( [ tf.reshape( self.features_h, [ -1, self.encoder_input_size ] ),
                                   tf.reshape( self.features_v, [ -1, self.encoder_input_size ] ) ], 1 )
      self.features_transposed = tf.concat(
        [tf.reshape(tf.transpose(self.features_h, [0, 1, 3, 2, 4]), [-1, self.encoder_input_size]),
         tf.reshape(self.features_v, [-1, self.encoder_input_size])], 1)
      self.encoder_nodes = self.features.shape.as_list() [1]


      # encode both h/v features into autoencoder bottleneck
      with tf.variable_scope( 'feature_encoder' ):
        for n in self.config.layers['autoencoder_nodes']:
          print( '  creating %i encoder nodes for dense layer.' % n )
          size_in = self.features.shape.as_list() [1]
          self.features = layers.bn_dense( self.features, size_in, n, self.phase, self.config.training, 'bn_feature_encoder_%i' % n )
          self.features_transposed = layers.bn_dense(self.features_transposed, size_in, n, self.phase,
                                                     self.config.training,
                                                     'bn_feature_encoder_%i' % n)
          self.encoder_nodes = self.features.shape.as_list() [1]

      with tf.variable_scope( 'input' ):
        self.input_features = tf.placeholder( tf.float32, self.features.shape.as_list() )

  #
  # CREATE DECODER LAYERS FOR 3D STREAMS
  # these decode into horizontal/vertical stacks,
  # with shared upconvolution kernels to save memory.
  #
  def create_3D_decoders( self ):
    for decoder_config in self.config.decoders_3D:
      with tf.device( '/device:GPU:%i' % ( decoder_config[ 'preferred_gpu' ] )):
        self.pinhole = False
        self.create_3D_decoder( decoder_config )
        self.pinhole = True
    if self.config.linked_decoders_3D:
      self.create_linked_3D_decoder(self.config.linked_decoders_3D)

  def create_3D_decoder( self, decoder_config ):

    decoder = dict()
    decoder_id = decoder_config[ 'id' ]
    print( 'creating decoder pipeline ' + decoder_id )

    # create a decoder pathway (center view only)
    with tf.variable_scope( decoder_id ):
      sh = self.feature_shape

      decoder[ 'id' ] = decoder_id
      decoder[ 'channels' ] = decoder_config[ 'channels' ]
      decoder[ 'loss_fn' ] = decoder_config[ 'loss_fn' ]
      decoder[ 'weight' ] = decoder_config[ 'weight' ]
      decoder[ 'train' ] = decoder_config[ 'train' ]
      decoder[ 'preferred_gpu' ] = decoder_config[ 'preferred_gpu' ]

      decoder[ 'nodes' ] = sh[2] * sh[3] * sh[4]
      decoder[ 'upconv_in' ] = layers.bn_dense( self.features, self.encoder_nodes, decoder['nodes'], self.phase, self.config.training, 'bn_decoder_' + decoder_id + '_in' )

      layout = self.config.layers['autoencoder']
      decoder[ 'layers_v' ] = []
      decoder[ 'layers_h' ] = []
      decoder[ 'variables' ] = []

      # decode features to upwards input using configured dense layers
      with tf.variable_scope( 'feature_decoder' ):
        num_layers = len( self.config.layers['autoencoder_nodes'] )
        if num_layers > 0:
          decoder[ 'features' ] = self.features
          size_list = reversed( self.config.layers['autoencoder_nodes'][0:-1] )
          for n in size_list:
            print( '  creating %i decoder nodes for dense layer.' % n )
            size_in = decoder[ 'features' ].shape.as_list() [1]
            decoder[ 'features' ] = layers.bn_dense( decoder[ 'features' ], size_in, n, self.phase, self.config.training, 'bn_feature_decoder_%i' % n )
            self.encoder_nodes = self.features.shape.as_list() [1]

          size_in = decoder[ 'features' ].shape.as_list() [1]
          decoder[ 'upconv_v' ] = layers.bn_dense( decoder[ 'features' ], size_in, self.encoder_input_size, self.phase, self.config.training, 'bn_upconv_v_in' )
          decoder[ 'upconv_v' ] = tf.reshape( decoder[ 'upconv_v' ], [ -1, sh[1], sh[2], sh[3], sh[4] ] )
          decoder[ 'upconv_h' ] = layers.bn_dense( decoder[ 'features' ], size_in, self.encoder_input_size, self.phase, self.config.training, 'bn_upconv_h_in' )
          decoder[ 'upconv_h' ] = tf.reshape( decoder[ 'upconv_h' ], [ -1, sh[1], sh[2], sh[3], sh[4] ] )

        else:
          # no dense encoder layers - wire directly
          decoder[ 'upconv_v' ] = self.features_v
          decoder[ 'upconv_h' ] = self.features_h


      # now the convolutional decoder layers
      last_layer = min( len(layout), self.max_layer )

      for i in range( 0, last_layer ):

        layer_id = "decoder_%s_%i" % (decoder_id, last_layer-i-1)
        print( '    generating upconvolution layer structure ' + layer_id )
        decoder[ 'variables' ].insert( i, layers.decoder_variables( layer_id, self.encoder_variables[-1-i] ))

        decoder[ 'layers_v' ].insert( i, layers.layer_upconv3d( layer_id + "_v",
                                                                decoder[ 'variables' ][i],
                                                                self.batch_size,
                                                                decoder[ 'upconv_v' ],
                                                                self.phase,
                                                                self.config.training ))
        if self.pinhole and i != last_layer - 1:
          decoder['upconv_v'] = tf.concat([decoder['layers_v'][-1].out, self.conv_layers_v[-2 - i].out], axis=4)
          decoder['variables'][i].pinhole_weight = layers.pinhole_weight(decoder['variables'][i], decoder['upconv_v'])
          decoder['upconv_v'] = layers.pinhole_conv3d(decoder['variables'][i], decoder['upconv_v'])
          print('pinhole')
        else:
          print('no pinhole')
          decoder['upconv_v'] = decoder['layers_v'][-1].out

        decoder[ 'layers_h' ].insert( i, layers.layer_upconv3d( layer_id + "_h",
                                                                decoder[ 'variables' ][i],
                                                                self.batch_size,
                                                                decoder[ 'upconv_h' ],
                                                                self.phase,
                                                                self.config.training ))
        if self.pinhole and i != last_layer - 1:
          decoder['upconv_h'] = tf.concat([decoder['layers_h'][-1].out, self.conv_layers_h[-2 - i].out], axis=4)
          decoder['upconv_h'] = layers.pinhole_conv3d(decoder['variables'][i], decoder['upconv_h'])
        else:
          decoder['upconv_h'] = decoder['layers_h'][-1].out

    # to save some memory, as we do not use it atm
    assert( loss_min_coord_3D == 0 )
    assert( loss_max_coord_3D == 48 )

    decoder['upconv_v_reduce'] = decoder['upconv_v'][ :, :, loss_min_coord_3D:loss_max_coord_3D, loss_min_coord_3D:loss_max_coord_3D, : ]
    decoder['upconv_h_reduce'] = decoder['upconv_h'][ :, :, loss_min_coord_3D:loss_max_coord_3D, loss_min_coord_3D:loss_max_coord_3D, : ]

    decoder['input_v'] = tf.placeholder( tf.float32, decoder['upconv_v'].shape.as_list() )
    decoder['input_h'] = tf.placeholder( tf.float32, decoder['upconv_h'].shape.as_list() )

    decoder['input_v_reduce'] = decoder['input_v'][ :, :, loss_min_coord_3D:loss_max_coord_3D, loss_min_coord_3D:loss_max_coord_3D, : ]
    decoder['input_h_reduce'] = decoder['input_h'][ :, :, loss_min_coord_3D:loss_max_coord_3D, loss_min_coord_3D:loss_max_coord_3D, : ]

    self.decoders_3D[decoder_id] = decoder


  def create_linked_3D_decoder(self, linked_decoders):
    decoder_ids = [linked_decoders[i]['id'] for i in range(0, len(linked_decoders))]
    assert (loss_min_coord_3D == 0)
    assert (loss_max_coord_3D == 48)

    for decoder_config in linked_decoders:
      with tf.device('/device:GPU:%i' % (decoder_config['preferred_gpu'])):
        decoder = dict()
        decoder_id = decoder_config['id']
        print('creating decoder pipeline ' + decoder_id)

        # create a decoder pathway (center view only)
        with tf.variable_scope(decoder_id):
          sh = self.feature_shape

          decoder['id'] = decoder_id
          decoder['channels'] = decoder_config['channels']
          decoder['loss_fn'] = decoder_config['loss_fn']
          decoder['weight'] = decoder_config['weight']
          decoder['train'] = decoder_config['train']
          decoder['preferred_gpu'] = decoder_config['preferred_gpu']
          decoder['nodes'] = sh[2] * sh[3] * sh[4]
          decoder['upconv_in'] = layers.bn_dense(self.features, self.encoder_nodes, decoder['nodes'], self.phase,
                                                 self.config.training, 'bn_decoder_' + decoder_id + '_in')

          decoder['layers_v'] = []
          decoder['layers_h'] = []
          decoder['variables'] = []

          # decode features to upwards input using configured dense layers
          with tf.variable_scope('feature_decoder'):
            num_layers = len(self.config.layers['autoencoder_nodes'])
            if num_layers > 0:
              decoder['features'] = self.features
              size_list = reversed(self.config.layers['autoencoder_nodes'][0:-1])
              for n in size_list:
                print('  creating %i decoder nodes for dense layer.' % n)
                size_in = decoder['features'].shape.as_list()[1]
                decoder['features'] = layers.bn_dense(decoder['features'], size_in, n, self.phase, self.config.training,
                                                      'bn_feature_decoder_%i' % n)
                self.encoder_nodes = self.features.shape.as_list()[1]

              size_in = decoder['features'].shape.as_list()[1]
              decoder['upconv_v'] = layers.bn_dense(decoder['features'], size_in, self.encoder_input_size, self.phase,
                                                    self.config.training, 'bn_upconv_v_in')
              decoder['upconv_v'] = tf.reshape(decoder['upconv_v'], [-1, sh[1], sh[2], sh[3], sh[4]])
              decoder['upconv_h'] = layers.bn_dense(decoder['features'], size_in, self.encoder_input_size, self.phase,
                                                    self.config.training, 'bn_upconv_h_in')
              decoder['upconv_h'] = tf.reshape(decoder['upconv_h'], [-1, sh[1], sh[2], sh[3], sh[4]])

            else:
              # no dense encoder layers - wire directly
              decoder['upconv_v'] = self.features_v
              decoder['upconv_h'] = self.features_h
          self.decoders_3D[decoder_id] = decoder
          del decoder
      # now the convolutional decoder layers
    layout = self.config.layers['autoencoder']
    last_layer = min(len(layout), self.max_layer)
    start_layer = 0
    for c_layer in self.config.connect_layers:
      for decoder_config in linked_decoders:
        decoder_id = decoder_config['id']
        with tf.device('/device:GPU:%i' % (decoder_config['preferred_gpu'])):
          for i in range(start_layer, c_layer):

            layer_id = "decoder_%s_%i" % (decoder_id, last_layer - i - 1)
            print('    generating upconvolution layer structure ' + layer_id)
            self.decoders_3D[decoder_id]['variables'].insert(i, layers.decoder_variables(layer_id, self.encoder_variables[-1 - i]))
            self.decoders_3D[decoder_id]['layers_v'].insert(i, layers.layer_upconv3d(layer_id + "_v",
                                                                                     self.decoders_3D[decoder_id][
                                                                                       'variables'][i],
                                                                                     self.batch_size,
                                                                                     self.decoders_3D[decoder_id][
                                                                                       'upconv_v'],
                                                                                     self.phase,
                                                                                     self.config.training,
                                                                                     shared_variables=True))

            if self.pinhole and i != c_layer - 1:
              self.decoders_3D[decoder_id]['upconv_v'] = tf.concat([self.decoders_3D[decoder_id]['layers_v'][-1].out,
                                                        self.conv_layers_v[-2 - i].out], axis=4)
              self.decoders_3D[decoder_id]['variables'][i].pinhole_weight = layers.pinhole_weight(self.decoders_3D[decoder_id]['variables'][i], self.decoders_3D[decoder_id]['upconv_v'])
              self.decoders_3D[decoder_id]['upconv_v'] = layers.pinhole_conv3d(self.decoders_3D[decoder_id]['variables'][i], self.decoders_3D[decoder_id]['upconv_v'])
            else:
              self.decoders_3D[decoder_id]['upconv_v'] = self.decoders_3D[decoder_id]['layers_v'][-1].out

            self.decoders_3D[decoder_id]['layers_h'].insert(i, layers.layer_upconv3d(layer_id + "_h",
                                                                                     self.decoders_3D[decoder_id][
                                                                                       'variables'][i],
                                                                                     self.batch_size,
                                                                                     self.decoders_3D[decoder_id][
                                                                                       'upconv_h'],
                                                                                     self.phase,
                                                                                     self.config.training,
                                                                                     shared_variables=True))
            if self.pinhole and i != c_layer - 1:
              self.decoders_3D[decoder_id]['upconv_h'] = tf.concat([self.decoders_3D[decoder_id]['layers_h'][-1].out,
                                                                    self.conv_layers_h[-2 - i].out], axis=4)
              self.decoders_3D[decoder_id]['upconv_h'] = layers.pinhole_conv3d(
                self.decoders_3D[decoder_id]['variables'][i], self.decoders_3D[decoder_id]['upconv_h'])

            else:
              self.decoders_3D[decoder_id]['upconv_h'] = self.decoders_3D[decoder_id]['layers_h'][-1].out

      # concatenate components
      start_layer = c_layer
      if start_layer != last_layer:
        joined_v = tf.concat([self.decoders_3D[id]['upconv_v'] for id in decoder_ids ], axis=4)
        joined_h = tf.concat([self.decoders_3D[id]['upconv_h'] for id in decoder_ids ], axis=4)
        if self.pinhole:
          joined_v = tf.concat([joined_v, self.conv_layers_v[-1 - c_layer].out], axis=4)
          joined_h = tf.concat([joined_h, self.conv_layers_h[-1 - c_layer].out], axis=4)

        for decoder_config in linked_decoders:
          decoder_id = decoder_config['id']
          with tf.device('/device:GPU:%i' % (decoder_config['preferred_gpu'])):
            with tf.variable_scope(decoder_id):
              self.decoders_3D[decoder_id]['variables'][c_layer-1].pinhole_weight = layers.pinhole_weight(
                self.decoders_3D[decoder_id]['variables'][c_layer-1],joined_v)

        for decoder_config in linked_decoders:
          decoder_id = decoder_config['id']
          with tf.device('/device:GPU:%i' % (decoder_config['preferred_gpu'])):
            with tf.variable_scope(decoder_id):
              self.decoders_3D[decoder_id]['upconv_v'] = layers.pinhole_conv3d( self.decoders_3D[decoder_id]['variables'][c_layer-1], joined_v)
              self.decoders_3D[decoder_id]['upconv_h'] = layers.pinhole_conv3d(self.decoders_3D[decoder_id]['variables'][c_layer-1], joined_h)

    for decoder_config in linked_decoders:
      decoder_id = decoder_config['id']
      with tf.device('/device:GPU:%i' % (decoder_config['preferred_gpu'])):
        self.decoders_3D[decoder_id]['upconv_v_reduce'] = self.decoders_3D[decoder_id][
          'upconv_v'][ :, :, loss_min_coord_3D:loss_max_coord_3D, loss_min_coord_3D:loss_max_coord_3D, : ]
        self.decoders_3D[decoder_id]['upconv_h_reduce'] = self.decoders_3D[decoder_id][
          'upconv_h'][ :, :, loss_min_coord_3D:loss_max_coord_3D, loss_min_coord_3D:loss_max_coord_3D, : ]

        self.decoders_3D[decoder_id]['input_v'] = tf.placeholder(tf.float32, self.decoders_3D[decoder_id]['upconv_v'].shape.as_list())
        self.decoders_3D[decoder_id]['input_h'] = tf.placeholder(tf.float32, self.decoders_3D[decoder_id]['upconv_h'].shape.as_list())

        self.decoders_3D[decoder_id]['input_v_reduce'] = self.decoders_3D[decoder_id][
          'input_v'][ :, :, loss_min_coord_3D:loss_max_coord_3D, loss_min_coord_3D:loss_max_coord_3D, : ]
        self.decoders_3D[decoder_id]['input_h_reduce'] = self.decoders_3D[decoder_id][
          'input_h'][ :, :, loss_min_coord_3D:loss_max_coord_3D, loss_min_coord_3D:loss_max_coord_3D, : ]


  #
  # CREATE DECODER LAYERS FOR ADDITIONAL DECODERS CONFIGURED IN THE CONFIG FILE
  #
  def create_2D_decoders( self ):
    for decoder_config in self.config.decoders_2D:
      with tf.device( '/device:GPU:%i' % ( decoder_config[ 'preferred_gpu' ] )):
        self.create_2D_decoder( decoder_config )


  def create_2D_decoder( self, decoder_config ):

    decoder = dict()
    decoder_id = decoder_config[ 'id' ]
    print( 'creating decoder pipeline ' + decoder_id )

    # create a decoder pathway (center view only)
    with tf.variable_scope( decoder_id ):
      sh = self.feature_shape

      decoder[ 'id' ] = decoder_id
      decoder[ 'channels' ] = decoder_config[ 'channels' ]
      decoder[ 'loss_fn' ] = decoder_config[ 'loss_fn' ]
      decoder[ 'weight' ] = decoder_config[ 'weight' ]
      decoder[ 'train' ] = decoder_config[ 'train' ]
      decoder[ 'preferred_gpu' ] = decoder_config[ 'preferred_gpu' ]

      # decode features to upwards input using configured dense layers
      num_layers = len( self.config.layers['2D_decoder_nodes'] )
      if num_layers > 0 and decoder[ 'loss_fn' ] != 'L2':
        decoder[ 'features' ] = self.features_transposed
        size_list = self.config.layers['2D_decoder_nodes']
        for n in size_list:
          print( '  creating %i decoder nodes for dense layer.' % n )
          size_in = decoder[ 'features' ].shape.as_list() [1]
          decoder[ 'features' ] = layers.bn_dense( decoder[ 'features' ], size_in, n, self.phase, self.config.training, 'bn_feature_decoder_%i' % n )
          self.encoder_nodes = self.features.shape.as_list() [1]

        size_in = decoder[ 'features' ].shape.as_list() [1]

        H = loss_max_coord_2D - loss_min_coord_2D
        W = loss_max_coord_2D - loss_min_coord_2D
        nodes = H * W * decoder[ 'channels']

        decoder[ 'upconv_reduce' ] = layers.bn_dense( decoder[ 'features' ], size_in, nodes, self.phase, self.config.training, 'bn_upconv' )
        decoder[ 'upconv_reduce' ] = tf.reshape( decoder[ 'upconv_reduce' ], [ -1, H,W, decoder[ 'channels' ]] )
        decoder['input'] = tf.placeholder( tf.float32, [ None, self.H, self.W, decoder['channels']] )
        decoder['input_reduce'] = decoder['input'][ :, loss_min_coord_2D:loss_max_coord_2D, loss_min_coord_2D:loss_max_coord_2D, : ]

      else:

        decoder[ 'nodes' ] = sh[2] * sh[3] * sh[4]
        decoder[ 'upconv_in' ] = layers.bn_dense( self.features_transposed, self.encoder_nodes, decoder['nodes'], self.phase, self.config.training, 'bn_decoder_' + decoder_id + '_in' )

        layout = self.config.layers['autoencoder']
        decoder[ 'layers' ] = []

        # decoding layer 0 - all variables already created
        sh = self.feature_shape
        decoder[ 'upconv' ] = tf.reshape( decoder[ 'upconv_in' ], [-1, sh[2], sh[3], sh[4] ] )
        last_layer = min( len(layout), self.max_layer )
        for i in range( 0, last_layer ):

          layer_id = "decoder_%s_%i" % (decoder_id, last_layer - i - 1)
          print( '    generating upconvolution layer structure ' + layer_id )

          out_channels = -1
          if i==last_layer-1:
            out_channels = decoder[ 'channels' ]

          # evil hack
          no_relu = False
          if i==last_layer-1 and decoder_id=='depth':
            print( '  skipping last ReLU for depth regression.' )
            no_relu = True

          decoder[ 'layers' ].insert( i,
            layers.layer_upconv2d( layer_id,
                                   self.encoder_variables[-1-i],
                                   self.batch_size,
                                   decoder[ 'upconv' ],
                                   self.phase,
                                   self.config.training,
                                   out_channels = out_channels,
                                   no_relu = no_relu  ))
          if self.pinhole and i != last_layer - 1:
            cv_pos = np.int32((self.conv_layers_h[-2 - i].output_shape[1] - 1) / 2)
            features_add = tf.concat([tf.transpose(self.conv_layers_h[-2 - i].out, [0, 1, 3, 2, 4])[:, cv_pos, :, :, :],
                                      self.conv_layers_v[-2 - i].out[:, cv_pos, :, :, :]], 3)
            decoder['upconv'] = tf.concat([decoder['layers'][-1].out, features_add], axis=3)
            decoder['layers'][i].pinhole_weight = layers.pinhole_weight_2d(decoder['layers'][i], decoder['upconv'])
            decoder['upconv'] = layers.pinhole_conv2d(decoder['layers'][i], decoder['upconv'])
          else:
            decoder['upconv'] = decoder['layers'][-1].out

        decoder['upconv_reduce'] = decoder['upconv'][ :, loss_min_coord_2D:loss_max_coord_2D, loss_min_coord_2D:loss_max_coord_2D, : ]

        decoder['input'] = tf.placeholder( tf.float32, [ None, self.H, self.W, decoder['channels']] )
        decoder['input_reduce'] = decoder['input'][ :, loss_min_coord_2D:loss_max_coord_2D, loss_min_coord_2D:loss_max_coord_2D, : ]

      self.decoders_2D[decoder_id] = decoder



  def add_training_ops( self ):

    print( 'creating training ops' )

    # what needs to be updated before training
    self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # L2-loss on feature layers

    for cfg in self.config.minimizers:

      minimizer = dict()
      minimizer[ 'id' ] = cfg[ 'id' ]
      print( '  minimizer ' + cfg[ 'id' ] )

      with tf.device( '/device:GPU:%i' % ( cfg[ 'preferred_gpu' ] )):
        minimizer[ 'loss' ] = 0
        minimizer[ 'requires' ] = []
        if 'losses_3D' in cfg:
          for id in cfg[ 'losses_3D' ]:

            # purely computational loss, no inputs
            if 'computational' in self.decoders_3D[id]:
              minimizer[ 'loss' ] += self.decoders_3D[id]['weight'] * self.decoders_3D[id]['loss']

            # for all others, add requirements
            elif self.decoders_3D[id][ 'train' ]:
              minimizer[ 'loss' ] += self.decoders_3D[id]['weight'] * (self.decoders_3D[id]['loss']+ self.decoders_3D[id]['loss_cv'])
              minimizer[ 'requires' ].append( id + '_v' )
              minimizer[ 'requires' ].append( id + '_h' )


        if 'losses_2D' in cfg:
          for id in cfg[ 'losses_2D' ]:
            if self.decoders_2D[id][ 'train' ]:
              minimizer[ 'loss' ] += self.decoders_2D[id]['weight'] * self.decoders_2D[id]['loss']
              minimizer[ 'requires' ].append( self.decoders_2D[id]['id'] )

        with tf.control_dependencies( self.update_ops ):
          # Ensures that we execute the update_ops before performing the train_step
          minimizer[ 'optimizer' ] = tf.train.AdamOptimizer( cfg[ 'step_size' ] )
          # gradient clipping
          gradients, variables = zip(
            *minimizer['optimizer'].compute_gradients(minimizer['loss'], colocate_gradients_with_ops=True))
          gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
          minimizer['train_step'] = minimizer['optimizer'].apply_gradients(zip(gradients, variables))

      self.minimizers[ cfg[ 'id' ] ] = minimizer



  # add training ops for additional decoder pathway (L2 loss)
  def setup_losses( self ):

    # loss function for auto-encoder
    for id in self.decoders_3D:
      with tf.device( '/device:GPU:%i' % ( self.decoders_3D[id][ 'preferred_gpu' ] )):

        # loss function for auto-encoder
        with tf.variable_scope( 'training_3D_' + id ):
          if self.decoders_3D[id][ 'loss_fn' ] == 'L2':
            print( '  creating L2-loss for decoder pipeline ' + id )
            self.decoders_3D[id]['loss_v'] = tf.losses.mean_squared_error( self.decoders_3D[id]['input_v_reduce'], self.decoders_3D[id]['upconv_v_reduce'] )
            self.decoders_3D[id]['loss_h'] = tf.losses.mean_squared_error( self.decoders_3D[id]['input_h_reduce'], self.decoders_3D[id]['upconv_h_reduce'] )
            self.decoders_3D[id]['loss'] = self.decoders_3D[id]['loss_v'] + self.decoders_3D[id]['loss_h']
            cv_mask = np.zeros([self.config.training['samples_per_batch'], self.D, self.H, self.W, self.C])
            cv_mask[:, 4, :, :, :] = 1
            self.decoders_3D[id]['loss_cv'] = tf.losses.mean_squared_error(self.decoders_3D[id]['upconv_v_reduce'],
                                                                           tf.transpose(
                                                                             self.decoders_3D[id]['upconv_h_reduce'],
                                                                             perm=[0, 1, 3, 2, 4]),
                                                                           weights=cv_mask)

          else:
            # not implemented
            assert( False )


    # hard-coded loss function
    # the diffuse + specular decoders should sum up to the input light field.
    if 'diffuse' in self.decoders_3D and 'specular' in self.decoders_3D:
      print( 'adding loss function for diffuse/specular sum' )
      with tf.device( '/device:GPU:0' ):
        sum_loss = dict()
        sum_loss['id'] = 'diffuse_specular_sum'
        sum_loss['sum_v'] = tf.losses.mean_squared_error( self.decoders_3D['diffuse']['upconv_v'] + self.decoders_3D['specular']['upconv_v'], self.stack_v )
        sum_loss['sum_h'] = tf.losses.mean_squared_error( self.decoders_3D['diffuse']['upconv_h'] + self.decoders_3D['specular']['upconv_h'], self.stack_h )
        sum_loss['loss']  = sum_loss[ 'sum_v' ] + sum_loss[ 'sum_h' ]
        sum_loss['computational'] = True
        sum_loss['weight'] = 1.0
        self.decoders_3D['diffuse_specular_sum'] = sum_loss

    for id in self.decoders_2D:
      # loss function for auto-encoder
      with tf.device( '/device:GPU:%i' % ( self.decoders_2D[id][ 'preferred_gpu' ] )):
        with tf.variable_scope( 'training_2D_' + id ):

          if self.decoders_2D[id][ 'loss_fn' ] == 'L2':
            print( '  creating L2-loss for decoder pipeline ' + id )
            self.decoders_2D[id]['loss'] = tf.losses.mean_squared_error( self.decoders_2D[id]['input_reduce'], self.decoders_2D[id]['upconv_reduce'] )

  # initialize new variables
  def initialize_uninitialized( self, sess ):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    
    for i in not_initialized_vars:
      print( str(i.name) )

    if len(not_initialized_vars):
      sess.run(tf.variables_initializer(not_initialized_vars))

  # prepare input
  def prepare_net_input( self, batch ):

      # default params for network input
      nsamples   = batch[ 'stacks_v' ].shape[0]

      # default params for network input
      net_in = {  self.input_features : np.zeros( [nsamples, self.encoder_nodes ], np.float32 ),
                  self.keep_prob:       1.0,
                  self.phase:           False,
                  self.noise_sigma:     self.config.training[ 'noise_sigma' ] }

      # bind 3D decoder inputs to batch stream
      for id in self.decoders_3D:
        decoder = self.decoders_3D[id]
        if 'input_v' in decoder and 'input_h' in decoder:

          if (id + '_v' in batch) and (id + '_h' in batch):
            if self.config.config['rgb']:
              net_in[ decoder['input_v'] ] = batch[id + '_v']
              net_in[ decoder['input_h'] ] = batch[id + '_h']
            else:
              net_in[ decoder['input_v'] ] = batch[id + '_v'].mean(4) [ :,:,:,:, np.newaxis ]
              net_in[ decoder['input_h'] ] = batch[id + '_h'].mean(4) [ :,:,:,:, np.newaxis ]
          else:
            sh = decoder['input_v'].shape.as_list()
            sh[0] = nsamples
            net_in[ decoder['input_v'] ] = np.zeros( sh, np.float32 )
            net_in[ decoder['input_h'] ] = np.zeros( sh, np.float32 )


      # bind 2D decoder inputs to batch stream
      for id in self.decoders_2D:

        # special treatment for depth (one-hot conversion might be necessary)
        decoder = self.decoders_2D[id]
        if 'input' in decoder:
          if id in batch:
            if self.config.config['rgb']:
              if len( batch[id].shape ) == 3:
                net_in[ decoder['input'] ] = batch[id] [ :,:,:, np.newaxis ]
              else:
                net_in[ decoder['input'] ] = batch[id]
          else:
            sh = decoder['input'].shape.as_list()
            sh[0] = nsamples
            net_in[ decoder['input'] ] = np.zeros( sh, np.float32 )


      # bind global input to stream
      if self.config.config['rgb']:
        net_in[ self.stack_v ] = batch[ 'stacks_v' ]
        net_in[ self.stack_h ] = batch[ 'stacks_h' ]
      else:
        net_in[ self.stack_v ] = batch[ 'stacks_v' ].mean(4) [:,:,:,:,np.newaxis]
        net_in[ self.stack_h ] = batch[ 'stacks_h' ].mean(4) [:,:,:,:,np.newaxis]

      return net_in
