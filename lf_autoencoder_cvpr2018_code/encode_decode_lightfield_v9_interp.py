#
# Push a light field through decoder/encoder modules of the autoencoder
#

from queue import Queue
import code
import numpy as np

import scipy
import scipy.signal

# timing and multithreading
import _thread
import time
from timeit import default_timer as timer

# light field GPU tools
import lf_tools
import libs.tf_tools as tft

# data config
import config_data_format as cdf
import interpolate_mask as im
import config_autoencoder_v9_final as hp

def add_result_to_cv( data, result, cv, LF_crosshair, mask_sum, bs_x, bs_y, bxx, dc ):

  """ note: numpy arrays are passed by reference ... I think
  """
  H_mask = hp.eval_res['h_mask']
  W_mask = hp.eval_res['w_mask']
  m = hp.eval_res['m']
  print( 'x', end='', flush=True )
  by = result[1]['py']
  sv = result[0]['cv']
  if len(sv.shape) !=4:
    sv_v = result[0][ 'cv_v']
    sv_h = result[0]['cv_h']

  mask = im.get_mask(H_mask,W_mask,m)
  mask3d = np.expand_dims(mask, axis = 2)
  mask3d = np.tile(mask3d, (1, 1, 3))
  maskLF = np.expand_dims(mask3d, axis = 3)
  maskLF = np.transpose(np.tile(maskLF, (1, 1, 1,9)),[3,0,1,2])

  # cv data is in the center of the result stack
  # lazy, hardcoded the current fixed size
  p = H_mask//2 - dc['SY']//2
  q = H_mask//2 + dc['SY']//2

  H_patch = dc['H']
  W_patch = dc['W']

  for bx in range(bxx):
    px = bs_x * bx + dc['SX']
    py = bs_y * by + dc['SY']
    if len(sv.shape) == 4:
      cv[py-p:py+q , px-p:px+q, :] = np.add(cv[py-p:py+q , px-p:px+q, : ],
      np.multiply(sv[bx, H_patch//2 - H_mask//2 : H_patch//2 + H_mask//2 ,H_patch//2 - H_mask//2 : H_patch//2 + H_mask//2, :], mask3d))
    else:
      cv[py - p:py + q, px - p:px + q, :] = np.add(cv[py - p:py + q, px - p:px + q, :], np.multiply(sv[bx, 4,
                                                                                                    H_patch // 2 - H_mask // 2: H_patch // 2 + H_mask // 2,
                                                                                                    H_patch // 2 - H_mask // 2: H_patch // 2 + H_mask // 2,
                                                                                                    :], mask3d))

      LF_crosshair[0, :, py - p:py + q, px - p:px + q, :] = np.add(LF_crosshair[0, :, py - p:py + q, px - p:px + q, :]
                                                                   , np.multiply(
          sv_v[bx, :, H_patch // 2 - H_mask // 2: H_patch // 2 + H_mask // 2,
          H_patch // 2 - H_mask // 2: H_patch // 2 + H_mask // 2, :], maskLF))

      LF_crosshair[1, :, py - p:py + q, px - p:px + q, :] = np.add(LF_crosshair[1, :, py - p:py + q, px - p:px + q, :]
                                                                   , np.multiply(
          sv_h[bx, :, H_patch // 2 - H_mask // 2: H_patch // 2 + H_mask // 2,
          H_patch // 2 - H_mask // 2: H_patch // 2 + H_mask // 2, :], maskLF))

    mask_sum[py-p:py+q , px-p:px+q] = mask_sum[py-p:py+q , px-p:px+q] + mask


def encode_decode_lightfield(data, LF, inputs, outputs, decoder_path='stacks', disp=None):
  # light field size
  H = LF.shape[2]
  W = LF.shape[3]
  dc = cdf.data_config

  # patch step sizes
  bs_y = dc['SY']
  bs_x = dc['SX']
  # patch height/width
  ps_y = dc['H']
  ps_x = dc['W']
  ps_v = dc['D']

  # patches per row/column
  by = np.int16((H - ps_y) / bs_y) + 1
  bx = np.int16((W - ps_x) / bs_x) + 1

  # one complete row per batch
  cv = np.zeros([H, W, 3], np.float32)
  mask_sum = np.zeros([H,W], dtype = np.float32)
  LF_crosshair = np.zeros([2,9,H,W,3], dtype = np.float32)

  print('starting LF encoding/decoding [', end='', flush=True)
  start = timer()

  results_received = 0
  for py in range(by):
    print('.', end='', flush=True)

    stacks_h = np.zeros([bx, ps_v, ps_y, ps_x, 3], np.float32)
    stacks_v = np.zeros([bx, ps_v, ps_y, ps_x, 3], np.float32)
    depth = np.zeros([bx, ps_y, ps_x], np.float32)

    for px in range(bx):
      # get single patch
      patch = cdf.get_patch(LF, cv, disp, py, px)

      stacks_v[px, :, :, :, :] = patch['stack_v']
      stacks_h[px, :, :, :, :] = patch['stack_h']
      depth[px, :, :] = patch['depth']

    # push complete batch to encoder/decoder pipeline
    batch = dict()
    batch['stacks_h'] = stacks_h
    batch['stacks_v'] = stacks_v
    batch['depth'] = depth
    batch['py'] = py
    batch['decoder_path'] = decoder_path

    inputs.put(batch)

    #
    if not outputs.empty():
      result = outputs.get()
      add_result_to_cv(data, result, cv, LF_crosshair, mask_sum, bs_x, bs_y, bx, dc)
      results_received += 1
      outputs.task_done()

  # catch remaining results
  while results_received < by:
    result = outputs.get()
    add_result_to_cv(data, result, cv, LF_crosshair, mask_sum, bs_x, bs_y, bx, dc)
    results_received += 1
    outputs.task_done()

  # elapsed time since start of dmap computation
  end = timer()
  total_time = end - start
  print('] done, total time %g seconds.' % total_time)

  # evaluate result
  mse = 0.0

  # compute stats and return result
  print('total time ', end - start)
  print('MSE          : ', mse)

  # code.interact( local=locals() )
  return (cv, total_time, mse, mask_sum, LF_crosshair)

def scale_back(im, mask):
  H = mask.shape[0]
  W = mask.shape[1]
  mask[mask == 0] = 1

  if len(im.shape) == 3:
    mask = np.expand_dims(mask, axis=2)
    mask = np.tile(mask, (1, 1, 3))

  if len(im.shape) == 5:
    mask = np.expand_dims(mask, axis=2)
    mask = np.tile(mask, (1, 1, 3))

    mask = np.expand_dims(mask, axis = 3)
    mask = np.transpose(np.tile(mask, (1, 1, 1, 9)), [3, 0, 1, 2])
    mask1 = np.zeros((2,9,H,W,3), dtype = np.float32)
    mask1[0,:,:,:,:] = mask
    mask1[1, :, :, :, :] = mask
    del mask
    mask = mask1

  return(np.divide(im,mask))
