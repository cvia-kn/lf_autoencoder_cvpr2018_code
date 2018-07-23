import numpy as np
from scipy import interpolate
import config_autoencoder_v9_final as hp

def get_mask(H,W,m):
    arr = np.zeros((H,W), dtype = np.float32) + hp.eval_res['min_mask']
    arr[1:H-1,1:W-1] = np.nan
    Hm = H//2
    Wm = W//2

    arr[Hm - m:Hm + m,Wm - m:Wm + m] = 1

    x = np.arange(0, arr.shape[1])
    y = np.arange(0, arr.shape[0])

    #mask invalid values
    arr = np.ma.masked_invalid(arr)
    xx, yy = np.meshgrid(x, y)
    #get only the valid values
    x1 = xx[~arr.mask]
    y1 = yy[~arr.mask]
    newarr = arr[~arr.mask]

    GD1 = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='cubic')

    return(GD1)