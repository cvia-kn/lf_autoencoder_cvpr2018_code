
from queue import Queue
import time
import numpy as np
import h5py

# plotting
import matplotlib.pyplot as plt

# timing and multithreading
import _thread

# light field GPU tools
import lf_tools

# evaluator thread
from encode_decode_lightfield_v9_interp import encode_decode_lightfield
from encode_decode_lightfield_v9_interp import scale_back
from thread_evaluate_v9 import evaluator_thread

# configuration
import config_autoencoder_v9_final as hp

# Model path setup
model_id = hp.network_model
model_path = './networks/' + model_id + '/model.ckpt'
result_folder = hp.eval_res['result_folder']
data_eval_folder = hp.eval_res['test_data_folder']

# I/O queues for multithreading
inputs = Queue( 15*15 )
outputs = Queue( 15*15 )

data_folders = (

# these data is in ./test_data folder
# ( "diffuse_specular", "intrinsic", "seen", "lf_test_intrinsic0cC7GPRFAIvP5i" ),
# ( "diffuse_specular",  "intrinsic", "not seen", "lf_test_intrinsic0AruXjjpWdmTOz" ),
# ( "diffuse_specular",  "benchmark", "seen", "lf_benchmark_cotton" ),
# ( "diffuse_specular",  "benchmark", "not seen", "lf_benchmark_bicycle" ),
# ( "diffuse_specular",  "benchmark", "not seen", "lf_benchmark_herbs" ),
# ( "diffuse_specular",  "lytro", "seen", "lf_test_lytro_cat3_lf.mat" ),
# ( "diffuse_specular",  "lytro", "not seen", "lf_test_lytro_buddha1.mat" ),
( "diffuse_specular",  "lytro", "not seen", "lf_test_lytro_IMG_2693_eslf.png.mat" ),

# these data you need to reproduce using scripts from DataRead_code

# ( "diffuse_specular", "intrinsic", "seen", "lf_test_intrinsicg3CxzfVmydmYGr" ),
# ( "diffuse_specular", "intrinsic", "seen", "lf_test_intrinsickqK4J4cafLswf2" ),
# ( "diffuse_specular",  "intrinsic", "not seen", "lf_test_intrinsicz1DefSIynpJhqi" ),
# ( "diffuse_specular",  "intrinsic", "not seen", "lf_test_intrinsiccTRQYxjW6XXw5J" ),
# ( "diffuse_specular",  "intrinsic", "not seen", "lf_test_intrinsicGumNhefYrATJLh" ),
# ( "diffuse_specular",  "intrinsic", "not seen", "lf_test_intrinsictrMltdlvXzRdOS" ),
# ( "diffuse_specular",  "intrinsic", "not seen", "lf_test_intrinsiciq16JtRgF7yzKp" ),
#
# ( "diffuse_specular",  "lytro", "seen", "lf_test_lytro_flowers_lf.mat" ),
# ( "diffuse_specular",  "lytro", "seen", "lf_test_lytro_koala1_lf.mat" ),
# ( "diffuse_specular",  "lytro", "seen", "lf_test_lytro_cat3_lf.mat" ),
# ( "diffuse_specular",  "lytro", "seen", "lf_test_lytro_hedgehog1_lf.mat" ),
# ( "diffuse_specular",  "lytro", "not seen", "lf_test_lytro_koala_lf.mat" ),
# ( "diffuse_specular",  "lytro", "not seen", "lf_test_lytro_buddha1.mat" ),
# ( "diffuse_specular",  "lytro", "not seen", "lf_test_lytro_IMG_2693_eslf.png.mat" ),
#
# ( "diffuse_specular",  "hci", "seen", "lf_test_hci_maria_lf.mat" ),
# ( "diffuse_specular",  "hci", "not seen", "lf_test_hci_cube_lf.mat" ),
#
# ( "diffuse_specular",  "benchmark", "not seen", "lf_benchmark_bicycle" ),
# ( "diffuse_specular",  "benchmark", "seen", "lf_benchmark_cotton" ),
# ( "diffuse_specular",  "benchmark", "seen", "lf_benchmark_antinous" ),
# ( "diffuse_specular",  "benchmark", "seen", "lf_benchmark_vinyl" ),
#
# ( "diffuse_specular", "stanford", "not seen", "lf_test_stanford_Amethyst_lf.mat" ),
# ( "diffuse_specular",  "stanford", "seen", "lf_test_stanford_LegoTruck_lf.mat" ),
#
#
# # numericat eval
# ( "diffuse_specular",  "intrinsic", "not seen", "lf_test_intrinsic0AruXjjpWdmTOz" ),
# ( "diffuse_specular",  "intrinsic", "not seen", "lf_test_intrinsic6CgoBrTon07emN" ),
# ( "diffuse_specular",  "intrinsic", "not seen", "lf_test_intrinsicBrZmxtWCIkYTFU" ),
# ( "diffuse_specular",  "intrinsic", "not seen", "lf_test_intrinsicDcO5nAshBnldAx" ),
# ( "diffuse_specular",  "intrinsic", "not seen", "lf_test_intrinsicF9oJj8EUagULX3" ),
# ( "diffuse_specular",  "intrinsic", "not seen", "lf_test_intrinsicFRMYJ3bYIKVICq" ),
)


# evaluator thread
_thread.start_new_thread( evaluator_thread,
                          ( model_path, hp, inputs,  outputs ))

# wait a bit to not skew timing results with initialization
time.sleep(20)

# loop over all datasets and collect errors
results = []
for lf_name in data_folders:
    file = h5py.File(result_folder + lf_name[3] + '.hdf5', 'w')
    if lf_name[1] == 'intrinsic':
    # stored directly in hdf5
        data_file = data_eval_folder + lf_name[0] + "/" + lf_name[1] + "/" + lf_name[2] + "/" + \
                lf_name[3] + ".hdf5"
        hdf_file = h5py.File( data_file, 'r')
        # hard-coded size, just for testing
        LF = hdf_file[ 'LF' ]
        cv_gt = lf_tools.cv( LF )

        LF_diffuse_gt = hdf_file[ 'LF_diffuse' ]
        diffuse_gt = lf_tools.cv( LF_diffuse_gt )

        LF_specular_gt = hdf_file[ 'LF_specular' ]
        specular_gt = lf_tools.cv( LF_specular_gt )

        disp_gt = hdf_file[ 'LF_disp' ]

        dmin = np.min(disp_gt)
        dmax = np.max(disp_gt)
    elif lf_name[1] == 'benchmark':
        data_file = data_eval_folder + lf_name[0] + "/" + lf_name[1] + "/" + lf_name[2] + "/" + \
                    lf_name[3] + ".hdf5"
        hdf_file = h5py.File(data_file, 'r')
        # hard-coded size, just for testing
        LF = hdf_file['LF']
        cv_gt = lf_tools.cv(LF)

        if lf_name[2] == "seen":
            disp_gt = hdf_file['LF_disp']

            dmin = np.min( disp_gt )
            dmax = np.max( disp_gt )
        else:
            dmin = -3.5
            dmax = 3.5
            disp_gt = np.zeros((cv_gt.shape[0], cv_gt.shape[1]), dtype=np.float32)

        diffuse_gt = np.zeros((cv_gt.shape[0],cv_gt.shape[1],3), dtype = np.float32)
        specular_gt = np.zeros((cv_gt.shape[0], cv_gt.shape[1], 3), dtype=np.float32)
    else:
        data_file = data_eval_folder + lf_name[0] + "/" + lf_name[1] + "/" + lf_name[2] + "/" + \
                    lf_name[3] + ".hdf5"
        hdf_file = h5py.File(data_file, 'r')
        # hard-coded size, just for testing
        LF = hdf_file['LF']
        cv_gt = lf_tools.cv(LF)
        disp_gt = np.zeros((cv_gt.shape[0],cv_gt.shape[1]), dtype = np.float32)
        diffuse_gt = np.zeros((cv_gt.shape[0],cv_gt.shape[1],3), dtype = np.float32)
        specular_gt = np.zeros((cv_gt.shape[0], cv_gt.shape[1], 3), dtype=np.float32)
        dmin = -3.5
        dmax = 3.5

    data = []
    result_cv = encode_decode_lightfield( data, LF,
                                        inputs, outputs,
                                        decoder_path='stacks',
                                        disp=disp_gt )

    cv_out = result_cv[0]
    mask = result_cv[3]
    LF_out = result_cv[4]

    cv_out = scale_back(cv_out, mask)
    LF_out = scale_back(LF_out, mask)

    test_decomposition = True

    if test_decomposition:
        result_diffuse = encode_decode_lightfield( data, LF,
                                                   inputs, outputs,
                                                   decoder_path='diffuse',
                                                   disp=disp_gt )
        result_specular = encode_decode_lightfield( data, LF,
                                                    inputs, outputs,
                                                    decoder_path='specular',
                                                    disp=disp_gt )
        diffuse_out = result_diffuse[0]
        LF_diffuse = result_diffuse[4]

        diffuse_out = scale_back(diffuse_out, mask)
        LF_diffuse = scale_back(LF_diffuse, mask)

        specular_out = result_specular[0]
        LF_specular = result_specular[4]

        specular_out = scale_back(specular_out, mask)
        LF_specular = scale_back(LF_specular, mask)

        cmin =   0.0
        cmax =  1.0
        cv = np.maximum( cmin, np.minimum( cmax, cv_out ))
        diffuse = np.maximum( cmin, np.minimum( cmax, diffuse_out ))
        specular = np.maximum( cmin, np.minimum( cmax, specular_out ))
        both = np.maximum( cmin, np.minimum( cmax, diffuse + specular ))

        # vertical stack
        plt.subplot(4, 2, 1)
        plt.imshow( np.clip(cv_gt,0,1) )

        # vertical stack
        plt.subplot(4, 2, 2)
        plt.imshow( np.clip(cv, 0,1) )

        # vertical stack center
        plt.subplot(4, 2, 3)
        plt.imshow( np.clip(diffuse_gt,0,1) )

        # horizontal stack center
        plt.subplot(4, 2, 4)
        plt.imshow( np.clip(diffuse, 0,1) )

        # vertical stack center
        plt.subplot(4, 2, 5)
        plt.imshow( np.clip(specular_gt, 0,1) )

        # horizontal stack center
        plt.subplot(4, 2, 6)
        plt.imshow( np.clip(specular, 0,1) )


        result_disp_regression = encode_decode_lightfield(data, LF,
                                               inputs, outputs,
                                               decoder_path='depth',
                                               disp=disp_gt)
        disp_regression_out = result_disp_regression[0]
        disp_regression_out = scale_back(disp_regression_out, mask)

        disp_regression = np.maximum(dmin, np.minimum(dmax, disp_regression_out[:,:,0]))

        # show estimated vs gt disp
        plt.subplot(4, 2, 7)
        plt.imshow(disp_gt)
        plt.subplot(4, 2, 8)
        plt.imshow( disp_regression )

        # compute deviation
        r = (result_cv[0] - (result_diffuse[0] + result_specular[0])) ** 2
        mse = np.sum(r) / np.float32( np.prod(r.shape) )
        print( 'MSE (cv vs. sum of components): %g' % mse )

        plt.show( block=False )

    dset_LF_out = file.create_dataset('LF_out', data = LF_out)
    dset_LF_diffuse = file.create_dataset('LF_diffuse', data = LF_diffuse)
    dset_LF_specular = file.create_dataset('LF_specular', data = LF_specular)
    dset_disp = file.create_dataset('disp', data=disp_regression)

inputs.put( () )

