# Wrapper class to read light field CNN training / test data
#
# (c) Bastian Goldluecke 4/2017, University of Konstanz
# License: Creative Commons BY-SA 4.0
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ctypes
import os
import h5py
import lf_tools

# dataset for light field angular patch stacks
class dataset:

  def __init__(self, filename, subsets={ 'validate' : 0.0, 'train': 1.0 }, max_examples=1e20, min_example=0, random_shuffle=True ):
    """Construct a light field angular patch dataset.
    Initialized from HDF5, which is expected to contain two datasets:
    'data'   - patch data, shape [num_examples, T,S, num_labels]
    'labels' - label data, shape [num_examples, 1]
    Labels are given in non-integers, i.e. can be between two integer labels.
    Labels are one-indexed (Matlab style)
    """

    print( 'reading HDF5 dataset ' + filename )
    self._filename = filename
    base=os.path.basename( filename )
    self._file_id = os.path.splitext(base)[0]
    self._file = h5py.File( self._filename,'r')
    self.streams = self._file.keys()
    self.stream = dict()
    self._num_examples = max_examples

    for stream in self.streams:
      print( '  found data stream ' + stream )
      self.stream[ stream ] = self._file[ stream ]
      sh = self.stream[ stream ].shape
      print( '    shape %s' % (sh,) )
      self._num_examples = min( self._num_examples, sh[ -1 ] )

    print( '  total %i training examples used.' % (self._num_examples) )

    seed = ctypes.c_ushort(hash( filename )).value
    np.random.seed( seed )

    self._permutation = np.arange( min_example, self._num_examples)
    if random_shuffle:
      np.random.shuffle( self._permutation )

    self.subsets = dict()
    min_index = 0
    for s in subsets:
      subset = dict()
      subset[ 'id' ]   = s

      # split off the index list for this subset from the global permutation
      p = subsets[s]
      n = int( p * self._num_examples )
      max_index = min( len(self._permutation), n+min_index )
      subset[ 'indices' ] = self._permutation[ min_index : max_index ]
      if random_shuffle:
        np.random.shuffle( subset[ 'indices' ] )

      # current subset epoch
      subset[ 'epoch' ] = 0
      # current offset and minibatch index
      subset[ 'index' ] = 0
      subset[ 'minibatch_index' ] = 0

      min_index = max_index
      self.subsets[ s ] = subset

    # flag if set is to be shuffled
    self._shuffle = random_shuffle


  # this function pulls exactly batch_size training examples for 
  # a certain subset of the training data.
  #
  # result will be None if subset is smaller than the batch size.
  #
  def next_batch( self, batch_size, subset_name='train' ):
    """Return the next `batch_size` examples from this data set."""

    subset = self.subsets[ subset_name ]
    subset[ 'minibatch_index' ] += 1

    # pull indices
    new_epoch = 0
    start = subset[ 'index' ]
    end = min( start + batch_size, len( subset[ 'indices' ] ))
    subset[ 'index' ] = end
    idx = subset[ 'indices' ][ start:end ]
    sz = len( idx )

    # if not enough examples drawn, reshuffle subset (if desired),
    # and pull remaining ones.
    missing = batch_size - sz
    if missing > 0:

      subset[ 'epoch' ] += 1
      subset[ 'index' ] = 0
      subset[ 'minibatch_index' ] = 0

      if self._shuffle:
        np.random.shuffle( subset[ 'indices' ] )

      start = subset[ 'index' ]
      end = min( start + missing, len( subset[ 'indices' ] ) )
      subset[ 'index' ] = end
      idx = np.append( idx, subset[ 'indices' ][ start:end ] )

    # if there are still samples missing, subset is too small
    sz = len( idx )
    if sz < batch_size:
      return None

    # retrieve index set from permuted index array
    batch = dict()
    for stream in self.streams:
      # create array for stream, requires stream shape
      sh = list( self.stream[ stream ].shape )
      batch[ stream ] = np.zeros( [sz] + sh[0:-1],  np.float32 )

    n = 0
    for i in idx:
      for stream in self.streams:
        sh = self.stream[ stream ].shape
        nd = len(sh) - 1
        batch_index = [ n ] + [slice(0,None)] * nd
        dataset_index = [slice(0,None)] * nd + [i]
        batch[ stream ][batch_index] = self.stream[ stream ][ tuple(dataset_index) ]

      if 'diffuse_v' in self.streams:
        batch = lf_tools.augment_data_intrinsic(batch, batch_index)
      else:
        batch = lf_tools.augment_data(batch, batch_index)
      n = n+1


    #code.interact( local = locals() )
    batch[ 'epoch' ] = subset[ 'epoch' ]
    batch[ 'minibatch_index' ] = subset[ 'minibatch_index' ]

    return batch


