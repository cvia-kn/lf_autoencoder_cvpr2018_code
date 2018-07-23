#!/usr/bin/python3

# start session
import os
import os.path
import code

import csv
from matplotlib.pyplot import *

log_path = "./networks/conv_data/current_v9/"
#history = 'validation'
history = 'training'
start_index = 400



# read log files
def read_file( filename ):

    entries = []

    losses = dict()
    log_index = -1
    training_run = -1
    
    with open( log_path + 'validation_history.csv', 'r') as csvfile:
        vread = csv.reader( csvfile, delimiter=',', quotechar='|' )
        for row in vread:
            entry = dict()
            entry[ 'timestamp ' ] = row[0]
            e_losses = dict()
            
            id = row[1]
            if id.isdigit():
                  # old format
                  entry[ 'feed' ]= int( id )
                  entry[ 'id' ] = 'unknown'
                  index = 2
            else:
                  entry[ 'id' ] = id
                  entry[ 'feed' ] = int( row[2] )
                  index = 3

            entry[ 'batch' ] = int( row[ index ] )
            index = index + 1
            entry[ 'epoch' ] = int( row[ index ] )
            index = index + 1

            if entry[ 'feed' ] == 0:
                # this starts a new log entry for all feeds
                log_index = log_index + 1
                if entry[ 'batch' ] == 0:
                    # restarted training
                    training_run = training_run + 1

            entry[ 'training_run' ] = training_run
            entry[ 'log_index' ] = log_index
            
            # parse loss values
            while index < len(row)-1:
                loss = row[ index ]
                number = row[ index+1 ]
                index = index + 2

                try:
                    number = float( number )
                    e_losses[ loss ] = float( number )
                    if not loss in losses:
                        losses[ loss ] = []

                    list = losses[ loss ]
                    while len( list ) < log_index+1:
                        list.append( [] )
                    list[ log_index ].append( ( entry['feed'], entry['id'], number ) )
                    
                except ValueError:
                    pass

            entry[ 'losses' ] = e_losses
            #print( entry )

            entries.append( entry )
            
    return entries, losses





# assemble a chart of a loss function, averaging over all datasets
def plot_loss( loss_name, history_name, datasets, window_size=1 ):

  labels = []
  data = []
  error = []

  loss = losses[ loss_name ]
  index = 0

  loss = loss[ start_index: ]
  for list in loss:
      test = []
      for t in list:
          if t[1] in datasets or len(datasets)==0:
              test.append( t[2] )

      #code.interact( local=locals() )
      data.append( np.mean( test ) )
      #data.append( np.median( test ) )      
      error.append( np.std( test ) )
      labels.append( index )
      index = index + 1


  # moving average
  data = np.convolve( data, np.ones(( window_size, )) / window_size, mode='valid' )
  error = np.convolve( error, np.ones(( window_size, )) / window_size, mode='valid' )
  labels = np.convolve( labels, np.ones(( window_size, )) / window_size, mode='valid' )

  # draw bar chars
  # error bars correspond to standard deviation over all data sets for this minibatch-group
  xlocations = np.array(range(len(data)))+0.5
  width = 0.5

  #bar(xlocations, data, yerr=error, width=width)
  bar( xlocations, data, yerr=error )

  #yticks(range(0, 8))
  #xticks(xlocations+ width/2, labels)
  #xlim(0, xlocations[-1]+width*2)
  title( "Loss %s on %s" % (loss_name, history_name) )

  gca().get_xaxis().tick_bottom()
  gca().get_yaxis().tick_left()
  
  show( block=False )




# MAIN
print( 'parsing validation log ...' )

entries, losses = read_file( ( log_path + '%s_history.csv' ) % history )
print( 'done.' )

window_size = 10


# datasets to be included in the plot
datasets_i = set()
datasets_i.add( 'lf_patch_autoencoder1' )
datasets_i.add( 'lf_patch_autoencoder2' )
datasets_i.add( 'lf_patch_autoencoder3' )
datasets_i.add( 'lf_patch_autoencoder4' )
datasets_i.add( 'lf_patch_autoencoder5' )
datasets_i.add( 'lf_patch_diffuse1' )
datasets_i.add( 'lf_patch_diffuse2' )
datasets_i.add( 'lf_patch_diffuse3' )

datasets_rw = set()
datasets_rw.add( 'lf_patch_lytro' )
datasets_rw.add( 'lf_patch_hci' )
datasets_rw.add( 'lf_patch_stanford' )

datasets_b = set()
datasets_b.add( 'lf_patch_autoencoder_depth_complete' )


# switch between different dataset groups for visualization
# can be empty, then all are averaged
datasets = set()
#datasets = datasets_b



figure(1)
plot_loss( 'stacks', history, datasets, window_size )

figure(2)
plot_loss( 'diffuse', history, datasets, window_size )

figure(3)
plot_loss( 'specular', history, datasets, window_size )

figure(4)
plot_loss( 'diffuse_specular_sum', history, datasets, window_size )

figure(5)
plot_loss( 'depth_regression', history, datasets, window_size )

figure(6)
plot_loss( 'depth', history, datasets, window_size )

code.interact( local = locals() )
