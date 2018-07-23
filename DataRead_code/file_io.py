#####################################################################
# This file is part of the 4D Light Field Benchmark.                #
#                                                                   #
# This work is licensed under the Creative Commons                  #
# Attribution-NonCommercial-ShareAlike 4.0 International License.   #
# To view a copy of this license,                                   #
# visit http://creativecommons.org/licenses/by-nc-sa/4.0/.          #
#####################################################################
#
# Note: (partially, what I needed) adapted to Python3
#


from six.moves import configparser
import os
import sys
import code
import imageio
import lf_tools

import numpy as np
import re


def read_lightfield(data_folder):
    print( 'Reading light field ', data_folder )
    params = read_parameters(data_folder)
    light_field = np.zeros((params["num_cams_x"], params["num_cams_y"], params["height"], params["width"], 3), dtype=np.float32)

    views = sorted([f for f in os.listdir(data_folder) if f.startswith("input_") and f.endswith(".png")])

    for idx, view in enumerate(views):
        fpath = os.path.join(data_folder, view)
        try:
            #print( fpath )
            img = read_img(fpath)
            light_field[ int(idx / params["num_cams_x"]), idx % params["num_cams_y"], :, :, :] = np.divide(img,255)
        except IOError:
            print( "Could not read input file: %s" ) #% fpath
            sys.exit()

    return light_field

def read_lightfield_crosshair(data_folder):
    print( 'Reading light field ', data_folder )
    params = read_parameters(data_folder)
    light_field = np.zeros((params["num_cams_x"], params["num_cams_y"], params["height"], params["width"], 3), dtype=np.float32)

    views = sorted([f for f in os.listdir(data_folder) if f.startswith("input_") and f.endswith(".png")])
    cam_pos = []
    for v in views:
        cam = re.search('input_Cam(.+?).png', v).group(1)
        cam_pos.append(int(cam))

    for idx, view in enumerate(views):
        fpath = os.path.join(data_folder, view)
        try:
            #print( fpath )
            img = read_img(fpath)
            light_field[ int(cam_pos[idx] / params["num_cams_x"]), cam_pos[idx] % params["num_cams_y"], :, :, :] = np.divide(img,255)
        except IOError:
            print( "Could not read input file: %s" ) #% fpath
            sys.exit()

    return light_field

def read_lightfield_intrinsic(data_folder, comp_name):
    print( 'Reading light field ', data_folder + ' component ', comp_name )
    params = read_parameters(data_folder)
    light_field = np.zeros((params["num_cams_x"], params["num_cams_y"], params["height"], params["width"], 3), dtype=np.float32)

    views = sorted([f for f in os.listdir(data_folder) if f.startswith(comp_name) and f.endswith(".exr")])

    for idx, view in enumerate(views):
        fpath = os.path.join(data_folder, view)
        try:
            #print( fpath )
            img = imageio.imread(fpath)
            light_field[ int(idx / params["num_cams_x"]), idx % params["num_cams_y"], :, :, :] = img
        except IOError:
            print( "Could not read input file: %s" ) #% fpath
            sys.exit()

    return light_field


def read_lightfield_intrinsic_crosshair(data_folder, comp_name):
    print( 'Reading light field ', data_folder + ' component ', comp_name )
    params = read_parameters(data_folder)
    light_field = np.zeros((params["num_cams_x"], params["num_cams_y"], params["height"], params["width"], 3), dtype=np.float32)

    views = sorted([f for f in os.listdir(data_folder) if f.startswith(comp_name) and f.endswith(".exr")])
    cam_pos = []
    for v in views:
        cam = re.search('Cam(.+?).exr', v).group(1)
        cam_pos.append(int(cam))

    for idx, view in enumerate(views):
        fpath = os.path.join(data_folder, view)
        try:
            #print( fpath )
            img = imageio.imread(fpath)
            light_field[ int(cam_pos[idx] / params["num_cams_x"]), cam_pos[idx] % params["num_cams_y"], :, :, :] = img
        except IOError:
            print( "Could not read input file: %s" ) #% fpath
            sys.exit()

    return light_field

def write_lightfield_intrinsic(data_folder, comp_name, factor):
    print( 'Writing light field ', data_folder + ' component ', comp_name )
    params = read_parameters(data_folder)

    views = sorted([f for f in os.listdir(data_folder) if f.startswith(comp_name) and f.endswith(".exr")])

    for idx, view in enumerate(views):
        fpath = os.path.join(data_folder, view)
        try:
            #print( fpath )
            img = imageio.imread(fpath)
            img = factor*img
            imageio.imwrite(fpath,img)
        except IOError:
            print( "Could not read input file: %s" ) #% fpath
            sys.exit()

def read_parameters(data_folder):
    params = dict()

    with open(os.path.join(data_folder, "parameters.cfg"), "r") as f:
        parser = configparser.ConfigParser()
        parser.readfp(f)

        section = "intrinsics"
        params["width"] = int(parser.get(section, 'image_resolution_x_px'))
        params["height"] = int(parser.get(section, 'image_resolution_y_px'))
        params["focal_length_mm"] = float(parser.get(section, 'focal_length_mm'))
        params["sensor_size_mm"] = float(parser.get(section, 'sensor_size_mm'))
        params["fstop"] = float(parser.get(section, 'fstop'))

        section = "extrinsics"
        params["num_cams_x"] = int(parser.get(section, 'num_cams_x'))
        params["num_cams_y"] = int(parser.get(section, 'num_cams_y'))
        params["baseline_mm"] = float(parser.get(section, 'baseline_mm'))
        params["focus_distance_m"] = float(parser.get(section, 'focus_distance_m'))
        params["center_cam_x_m"] = float(parser.get(section, 'center_cam_x_m'))
        params["center_cam_y_m"] = float(parser.get(section, 'center_cam_y_m'))
        params["center_cam_z_m"] = float(parser.get(section, 'center_cam_z_m'))
        params["center_cam_rx_rad"] = float(parser.get(section, 'center_cam_rx_rad'))
        params["center_cam_ry_rad"] = float(parser.get(section, 'center_cam_ry_rad'))
        params["center_cam_rz_rad"] = float(parser.get(section, 'center_cam_rz_rad'))

        section = "meta"
        params["disp_min"] = float(parser.get(section, 'disp_min'))
        params["disp_max"] = float(parser.get(section, 'disp_max'))
        params["frustum_disp_min"] = float(parser.get(section, 'frustum_disp_min'))
        params["frustum_disp_max"] = float(parser.get(section, 'frustum_disp_max'))
        params["depth_map_scale"] = float(parser.get(section, 'depth_map_scale'))

        params["scene"] = parser.get(section, 'scene')
        params["category"] = parser.get(section, 'category')
        params["date"] = parser.get(section, 'date')
        params["version"] = parser.get(section, 'version')
        params["authors"] = parser.get(section, 'authors').split(", ")
        params["contact"] = parser.get(section, 'contact')

    return params


def read_depth(data_folder, highres=False):
    fpath = os.path.join(data_folder, "gt_depth_%s.pfm" % ("highres" if highres else "lowres"))
    try:
        data = read_pfm(fpath)
    except IOError:
        print( "Could not read depth file: %s" )#% fpath
        sys.exit()
    return data


def read_disparity(data_folder, highres=False):
    fpath = os.path.join(data_folder, "gt_disp_%s.pfm" % ("highres" if highres else "lowres"))
    try:
        data = read_pfm(fpath)
    except IOError:
        print( "Could not read disparity file: %s" %( data_folder ))
        params = read_parameters(data_folder)
        data = ( np.zeros( [ params[ "height" ], params[ "width" ] ], np.float32 ), 1.0 )

    return data

def read_disparity_crosshair(data_folder):
    print( 'Reading light field ', data_folder + ' disparity ' )
    params = read_parameters(data_folder)
    disparity_lf = np.zeros((params["num_cams_x"], params["num_cams_y"], params["height"], params["width"]), dtype=np.float32)

    views = sorted([f for f in os.listdir(data_folder) if f.startswith('gt_disp_lowres_Cam') and f.endswith(".pfm")])
    cam_pos = []
    for v in views:
        cam = re.search('Cam(.+?).pfm', v).group(1)
        cam_pos.append(int(cam))

    for idx, view in enumerate(views):
        fpath = os.path.join(data_folder, view)
        try:
            data = read_pfm(fpath)
            data = np.array( data[0] )
            data = np.flip( data,0 )
            disparity_lf[ int(cam_pos[idx] / params["num_cams_x"]), cam_pos[idx] % params["num_cams_y"], :, :] = data
        except IOError:
            print( "Could not read input file: %s" ) #% fpath
            sys.exit()

    return disparity_lf

def read_normals_crosshair(data_folder):
    print( 'Reading normals ', data_folder )
    params = read_parameters(data_folder)
    light_field = np.zeros((params["num_cams_x"], params["num_cams_y"], params["height"], params["width"], 3), dtype=np.float32)

    views = sorted([f for f in os.listdir(data_folder) if f.startswith("normals_Cam") and f.endswith(".exr")])
    cam_pos = []
    for v in views:
        cam = re.search('normals_Cam(.+?).exr', v).group(1)
        cam_pos.append(int(cam))

    for idx, view in enumerate(views):
        fpath = os.path.join(data_folder, view)
        try:
            # print( fpath )
            normals = imageio.imread(fpath)
            normals[:, :, 1] = - normals[:, :, 1]
            normals[:, :, 2] = - normals[:, :, 2]
            light_field[int(cam_pos[idx] / params["num_cams_x"]), cam_pos[idx] % params["num_cams_y"], :, :,
            :] = normals
        except IOError:
            print("Could not read input file: %s")  # % fpath
            sys.exit()

    return light_field

def read_normals(data_folder):
    print( 'Reading normals ', data_folder )
    params = read_parameters(data_folder)
    light_field = np.zeros((params["num_cams_x"], params["num_cams_y"], params["height"], params["width"], 3), dtype=np.float32)

    view = 'normals_Cam040.exr'
    fpath = os.path.join(data_folder, view)
    try:
        normals = imageio.imread(fpath)
    except IOError:
        print( "Could not read input file: %s" ) #% fpath
        sys.exit()

    return normals

def read_img(fpath):
    from scipy import misc
    data = misc.imread(fpath)
    return data


def write_hdf5(data, fpath):
    import h5py
    h = h5py.File(fpath, 'w')
    for key, value in data.iteritems():
        h.create_dataset(key, data=value)
    h.close()


def write_pfm(data, fpath, scale=1, file_identifier="Pf", dtype="float32"):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

    data = np.flipud(data)
    height, width = np.shape(data)[:2]
    values = np.ndarray.flatten(np.asarray(data, dtype=dtype))
    endianess = data.dtype.byteorder
    print( endianess )

    if endianess == '<' or (endianess == '=' and sys.byteorder == 'little'):
        scale *= -1

    with open(fpath, 'wb') as file:
        file.write(file_identifier + '\n')
        file.write('%d %d\n' % (width, height))
        file.write('%d\n' % scale)
        file.write(values)




        
import numpy as np
import re
import sys


'''
Load a PFM file into a Numpy array. Note that it will have
a shape of H x W, not W x H. Returns a tuple containing the
loaded image and the scale factor from the file.
'''
def read_pfm( filename ):
  color = None
  width = None
  height = None
  scale = None
  endian = None

  file = open( filename, 'r', encoding='ISO-8859-1' )
  header = file.readline().rstrip()
  if header == 'PF':
    color = True    
  elif header == 'Pf':
    color = False
  else:
    raise Exception('Not a PFM file.')

  dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
  if dim_match:
    width, height = map(int, dim_match.groups())
  else:
    raise Exception('Malformed PFM header.')

  scale = float(file.readline().rstrip())
  if scale < 0: # little-endian
    endian = '<'
    scale = -scale
  else:
    endian = '>' # big-endian

  data = np.fromfile(file, endian + 'f')
  shape = (height, width, 3) if color else (height, width)
  return np.reshape(data, shape), scale



'''
Save a Numpy array to a PFM file.
'''
def write_pfm(filename, image, scale = 1):

  file = open( filename, 'w', encoding='ISO-8859-1' )
  image = np.flipud(image).astype( np.float32 )
  color = None

  if image.dtype.name != 'float32':
    raise Exception('Image dtype must be float32.')

  if len(image.shape) == 3 and image.shape[2] == 3: # color image
    color = True
  elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
    color = False
  else:
    raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

  file.write('PF\n' if color else 'Pf\n')
  file.write('%d %d\n' % (image.shape[1], image.shape[0]))

  endian = image.dtype.byteorder

  if endian == '<' or endian == '=' and sys.byteorder == 'little':
    scale = -scale

  file.write('%f\n' % scale)
  image.tofile(file)


