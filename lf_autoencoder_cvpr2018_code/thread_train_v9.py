#
# Thread for training the autoencoder network.
# Idea is that we can stream data in parallel, which is a bottleneck otherwise
#
import code
import os
import datetime
import sys

import numpy as np

import tensorflow as tf
import libs.tf_tools as tft
import config_data_format as cdf


def trainer_thread( model_path, hp, inputs ):

  # start tensorflow session
  session_config = tf.ConfigProto( allow_soft_placement=True,
                                   log_device_placement=hp.config[ 'log_device_placement' ] )
  sess = tf.InteractiveSession( config=session_config )

  # import network
  from   cnn_autoencoder_v9 import create_cnn
  cnn = create_cnn( hp )

  # add optimizers (will be saved with the network)
  cnn.add_training_ops()
  # start session
  print( '  initialising TF session' )
  sess.run(tf.global_variables_initializer())
  print( '  ... done' )

  # save object
  print( '  checking for model ' + model_path )
  if os.path.exists( model_path + 'model.ckpt.index' ):
    print( '  restoring model ' + model_path )
    tft.optimistic_restore( sess,  model_path + 'model.ckpt' )
    print( '  ... done.' )
  else:
    print( '  ... not found.' )

  # new saver object with complete network
  saver = tf.train.Saver()


  # statistics
  count = 0.0
  print( 'lf cnn trainer waiting for inputs' )


  terminated = 0;
  while not terminated:

    batch = inputs.get()
    if batch == ():
            terminated = 1
    else:

      niter      = batch[ 'niter' ]
      ep         = batch[ 'epoch' ]

      # default params for network input
      net_in = cnn.prepare_net_input( batch )

      # evaluate current network performance on mini-batch
      if batch[ 'logging' ]:

        print()
        sys.stdout.write( '  dataset(%d:%s) ep(%d) batch(%d) : ' %(batch[ 'nfeed' ], batch[ 'feed_id' ], ep, niter) )

        #loss_average = (count * loss_average + loss) / (count + 1.0)
        count = count + 1.0
        fields=[ '%s' %( datetime.datetime.now() ), batch[ 'feed_id' ], batch[ 'nfeed' ], niter, ep ]

        # compute loss for decoder pipelines
        for id in cnn.decoders_3D:
          if id + '_v' in batch or 'computational' in cnn.decoders_3D[id]:
            ( loss ) = sess.run( cnn.decoders_3D[id]['loss'], feed_dict=net_in )
            sys.stdout.write( '  %s %g   ' %(id, loss) )
            if not 'computational' in cnn.decoders_3D[id]:
              (loss_cv) = sess.run(cnn.decoders_3D[id]['loss_cv'], feed_dict=net_in)
              sys.stdout.write('  %s %g   ' % (id + '_cv', loss_cv))
            fields.append( id )
            fields.append( loss )
          else:
            fields.append( '' )
            fields.append( '' )

        for id in cnn.decoders_2D:
          if id in batch:
            ( loss ) = sess.run( cnn.decoders_2D[id]['loss'], feed_dict=net_in )
            sys.stdout.write( '  %s %g   ' %(id, loss) )
            fields.append( id )
            fields.append( loss )
          else:
            fields.append( '' )
            fields.append( '' )

        import csv
        with open( model_path + batch[ 'logfile' ], 'a+') as f:
          writer = csv.writer(f)
          writer.writerow(fields)

        print( '' )
        #code.interact( local=locals() )


      if batch[ 'niter' ] % hp.training[ 'save_interval' ] == 0 and niter != 0 and batch[ 'nfeed' ] == 0 and batch[ 'training' ]:
        # epochs now take too long, save every few 100 steps
        # Save the variables to disk.
        save_path = saver.save(sess, model_path + 'model.ckpt' )
        print( 'NEXT EPOCH' )
        print("  model saved in file: %s" % save_path)
        # statistics
        #print("  past epoch average loss %g"%(loss_average))
        count = 0.0


      # run training step
      if batch[ 'training' ]:
        net_in[ cnn.phase ] = True
        #code.interact( local=locals() )
        sys.stdout.write( '.' ) #T%i ' % int(count) )
        for id in cnn.minimizers:
          # check if all channels required for minimizer are present in batch
          ok = True
          for r in cnn.minimizers[id][ 'requires' ]:
            if not (r in batch):
              ok = False

          if ok:
            sys.stdout.write( cnn.minimizers[id][ 'id' ] + ' ' )
            sess.run( cnn.minimizers[id][ 'train_step' ],
                      feed_dict = net_in )

        sys.stdout.flush()

    inputs.task_done()
