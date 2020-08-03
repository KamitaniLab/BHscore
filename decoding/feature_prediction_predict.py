'''Feature prediction: prediction (test) script'''


from __future__ import print_function

import glob
import os
import argparse
from itertools import product
from time import time
import hdf5storage

import numpy as np

import bdpy
from bdpy.ml import ModelTest
from bdpy.dataform import Features, save_array, load_array
from bdpy.distcomp import DistComp
from bdpy.util import makedir_ifnot, get_refdata, dump_info

from fastl2lir import FastL2LiR


# Settings ###################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--net', '-n', default=None, help='Target deep neural network (e.g., "caffe/AlexNet")')
args = parser.parse_args()

# Network name
network = None

if args.net is not None:
    network = args.net

if network is None:
    raise RuntimeError('Target deep neural network is not specified. Please give "--net" option to the script or specify the networn in the script ("network").')

# Brain data
brain_dir = '../data/fmri'
subjects_list = {
    'sub-01':  'sub-01_perceptionNaturalImageTest_original_VC.h5',
    'sub-02':  'sub-02_perceptionNaturalImageTest_original_VC.h5',
    'sub-03':  'sub-03_perceptionNaturalImageTest_original_VC.h5',
}


rois_list = {
    'HVC' : 'ROI_HVC = 1',
    'V1'  : 'ROI_V1 = 1',
    'V2'  : 'ROI_V2 = 1',
    'V3'  : 'ROI_V3 = 1',
    'V4'  : 'ROI_V4 = 1',
}

# Image features
features_dir = '../data/features/ImageNetTest'
features_list = [d for d in os.listdir(os.path.join(features_dir, network)) if os.path.isdir(os.path.join(features_dir, network, d))]  # All layers
print('DNN feature')
print(os.path.join(features_dir, network))
print('Layers')
print(features_list)
features_list = features_list[::-1] # Start decoding from deep layers

# Trained models
models_dir_root = '../data/feature_decoders/ImageNetTraining'

# Results directory
results_dir_decoded_features_root = '../data/decoded_features/ImageNetTest'
results_dir_decoding_accuracy_root = '../data/decoding_accuracy/ImageNetTest'

# Misc settings
chunk_axis = None
# The features were divided into chunks along chunk_axis in decoder training.

label_name = 'image_index'

# Main #######################################################################

analysis_basename = os.path.splitext(os.path.basename(__file__))[0]

# Load data --------------------------------------------------------
print('----------------------------------------')
print('Loading data')

data_brain = {sbj: bdpy.BData(os.path.join(brain_dir, dat_file))
              for sbj, dat_file in subjects_list.items()}
data_features = Features(os.path.join(features_dir, network))

# Initialize directories -------------------------------------------
makedir_ifnot(os.path.join(results_dir_decoded_features_root, network))
makedir_ifnot(os.path.join(results_dir_decoding_accuracy_root, network))
makedir_ifnot('tmp')

# Save runtime information -----------------------------------------
runtime_params = {
    'fMRI data'                : [os.path.abspath(os.path.join(brain_dir, v)) for v in subjects_list.values()],
    'ROIs'                     : rois_list.keys(),
    'feature_decoders'         : os.path.abspath(models_dir_root),
    'target DNN'               : network,
    'target DNN features'      : os.path.abspath(os.path.join(features_dir, network)),
    'target DNN layers'        : features_list,
}
dump_info(os.path.join(results_dir_decoded_features_root, network), script=__file__, parameters=runtime_params)
dump_info(os.path.join(results_dir_decoding_accuracy_root, network), script=__file__, parameters=runtime_params)

# Analysis loop ----------------------------------------------------
print('----------------------------------------')
print('Analysis loop')

for feat, sbj, roi in product(features_list, subjects_list, rois_list):
    print('--------------------')
    print('Feature:    %s' % feat)
    print('Subject:    %s' % sbj)
    print('ROI:        %s' % roi)

    # Distributed computation setup
    # -----------------------------
    analysis_id = analysis_basename + '-' + sbj + '-' + roi + '-' + feat
    results_dir_prediction = os.path.join(results_dir_decoded_features_root, network, feat, sbj, roi)
    results_dir_accuracy = os.path.join(results_dir_decoding_accuracy_root, network, feat, sbj, roi)

    if os.path.exists(results_dir_prediction):
        print('%s is already done. Skipped.' % analysis_id)
        continue

    distcomp_db = os.path.join('./tmp', analysis_basename + '.db')
    distcomp = DistComp(backend='sqlite3', db_path=distcomp_db)
    if not distcomp.lock(analysis_id):
        print('%s is already running. Skipped.' % analysis_id)
        continue

    # Preparing data
    # --------------
    print('Preparing data')

    start_time = time()

    # Brain data
    x = data_brain[sbj].select(rois_list[roi])               # Brain data
    x_labels = data_brain[sbj].select(label_name).flatten()  # Image labels in the brain data

    # Target features and image labels (file names)
    y = data_features.get_features(feat)
    y_labels = data_features.index

    # Averaging brain data
    x_labels_unique = np.unique(x_labels)
    x = np.vstack([np.mean(x[(np.array(x_labels) == lb).flatten(), :], axis=0) for lb in x_labels_unique])

    print('Elapsed time (data preparation): %f' % (time() - start_time))

    # Model directory
    # ---------------
    model_dir = os.path.join(models_dir_root, network, feat, sbj, roi, 'model')

    # Preprocessing
    # -------------
    x_mean = load_array(os.path.join(model_dir, 'x_mean.mat'), key='x_mean')  # shape = (1, n_voxels)
    x_norm = load_array(os.path.join(model_dir, 'x_norm.mat'), key='x_norm')  # shape = (1, n_voxels)
    y_mean = load_array(os.path.join(model_dir, 'y_mean.mat'), key='y_mean')  # shape = (1, shape_features)
    y_norm = load_array(os.path.join(model_dir, 'y_norm.mat'), key='y_norm')  # shape = (1, shape_features)

    x = (x - x_mean) / x_norm

    # Prediction
    # ----------
    print('Prediction')

    start_time = time()

    model = FastL2LiR()

    test = ModelTest(model, x)
    test.model_format = 'bdmodel'
    test.model_path = model_dir
    test.dtype = np.float32
    test.chunk_axis = chunk_axis

    y_pred = test.run()

    print('Total elapsed time (prediction): %f' % (time() - start_time))

    # Postprocessing
    # --------------
    y_pred = y_pred * y_norm + y_mean

    # Calculate prediction accuracy
    # -----------------------------
    print('Prediction accuracy')

    start_time = time()

    y_pred_2d = y_pred.reshape([y_pred.shape[0], -1])
    y_true_2d = y.reshape([y.shape[0], -1])

    y_true_2d = get_refdata(y_true_2d, np.array(y_labels), x_labels_unique)

    n_units = y_true_2d.shape[1]

    accuracy = np.array([np.corrcoef(y_pred_2d[:, i].flatten(), y_true_2d[:, i].flatten())[0, 1]
                         for i in range(n_units)])
    accuracy = accuracy.reshape((1,) + y_pred.shape[1:])

    print('Total elapsed time (prediction accuracy): %f' % (time() - start_time))

    # Save results
    # ------------
    print('Saving results')

    makedir_ifnot(results_dir_prediction)
    makedir_ifnot(results_dir_accuracy)

    start_time = time()

    # Predicted features
    for i, label in enumerate(x_labels_unique):
        # Predicted features
        feat = np.array([y_pred[i,]])  # To make feat shape 1 x M x N x ...

        # Save file name
        save_file = os.path.join(results_dir_prediction, '%s.mat' % label)

        # Save
        save_array(save_file, feat, key='feat', dtype=np.float32, sparse=False)

    print('Saved %s' % results_dir_prediction)

    # Prediction accuracy
    save_file = os.path.join(results_dir_accuracy, 'accuracy.mat')
    save_array(save_file, accuracy, key='accuracy', dtype=np.float32, sparse=False)
    print('Saved %s' % save_file)

    print('Elapsed time (saving results): %f' % (time() - start_time))

    distcomp.unlock(analysis_id)

print('%s finished.' % analysis_basename)
