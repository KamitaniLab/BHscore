'''Feature prediction: decoders training script created by Shuntaro Aoki, modified by Soma Nonaka'''


from __future__ import print_function

import os
import warnings
import yaml
import json
from itertools import product
from time import time

import numpy as np

import bdpy
from bdpy.ml import ModelTraining
from bdpy.dataform import Features, save_array
from bdpy.distcomp import DistComp
from bdpy.util import makedir_ifnot, dump_info

from fastl2lir import FastL2LiR


# Settings ###################################################################

# network name
network = <Put your network name here>

# Brain data
brain_dir = '../data'
subjects_list = {
    'sub-01':  'sub-01_perceptionNaturalImageTraining_original_VC.h5',
    'sub-02':  'sub-02_perceptionNaturalImageTraining_original_VC.h5',
    'sub-03':  'sub-03_perceptionNaturalImageTraining_original_VC.h5',
}

label_name = 'stimulus_name'

rois_list = {
    'HVC': 'ROI_HVC = 1',
    'V1':  'ROI_V1 = 1',
    'V2':  'ROI_V2 = 1',
    'V3':  'ROI_V3 = 1',
    'V4':  'ROI_hV4 = 1',
}

num_voxel = {
    'HVC': 500,
    'V1':  500,
    'V2':  500,
    'V3':  500,
    'V4':  500,
}

# Image features
features_dir = '../data/features'
features_list = [d for d in os.listdir(os.path.join(features_dir, network)) if os.path.isdir(os.path.join(features_dir, network, d))]  # All layers
print('DNN feature')
print(os.path.join(features_dir, network))
features_list = features_list[::-1]  # Start training from deep layers
feature_index_file = 'index_random1000.mat'

# Model parameters
alpha = 100

# number of units to predict
n_sample = 1000

# Results directory
results_dir_root = os.path.join('./results/feature_decoders/', network)

# Misc settings
chunk_axis = None
# If Y.ndim >= 3, Y is divided into chunks along `chunk_axis`.
# Note that Y[0] should be sample dimension.


# Main #######################################################################

analysis_basename = os.path.splitext(os.path.basename(__file__))[0]

# Print info -----------------------------------------------------------------
print('Subjects:        %s' % subjects_list.keys())
print('ROIs:            %s' % rois_list.keys())
print('Target features: %s' % network)
print('Layers:          %s' % features_list)
print('')

# Load data ------------------------------------------------------------------
print('----------------------------------------')
print('Loading data')

data_brain = {sbj: bdpy.BData(os.path.join(brain_dir, dat_file))
              for sbj, dat_file in subjects_list.items()}
data_features = Features(os.path.join(
    features_dir, network))

# Initialize directories -----------------------------------------------------
makedir_ifnot(results_dir_root)
makedir_ifnot(os.path.join(results_dir_root, network))
makedir_ifnot('tmp')

# Save runtime information ---------------------------------------------------
info_dir = os.path.join(results_dir_root, network)
runtime_params = {
    'learning method':          'PyFastL2LiR',
    'regularization parameter': alpha,
    'fMRI data':                [os.path.abspath(os.path.join(brain_dir, v)) for v in subjects_list.values()],
    'ROIs':                     rois_list.keys(),
    'target DNN':               network,
    'target DNN features':      os.path.abspath(os.path.join(features_dir, network)),
    'target DNN layers':        features_list,
}
dump_info(info_dir, script=__file__, parameters=runtime_params)

# Analysis loop --------------------------------------------------------------
print('----------------------------------------')
print('Analysis loop')

loaded_features = []
for feat, sbj, roi in product(features_list, subjects_list, rois_list):
    print('--------------------')
    print('Feature:    %s' % feat)
    print('Subject:    %s' % sbj)
    print('ROI:        %s' % roi)
    print('Num voxels: %d' % num_voxel[roi])

    # Setup
    # -----
    analysis_id = analysis_basename + '-' + sbj + '-' + roi + '-' + feat
    results_dir = os.path.join(
        results_dir_root, network, feat, sbj, roi, 'model')
    makedir_ifnot(results_dir)

    # Check whether the analysis has been done or not.
    info_file = os.path.join(results_dir, 'info.yaml')
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            info = yaml.load(f)
        if '_status' in info and 'computation_status' in info['_status']:
            if info['_status']['computation_status'] == 'done':
                print('%s is already done and skipped' % analysis_id)
                continue

    # Preparing data
    # --------------
    print('Preparing data')

    start_time = time()

    # Brain data
    x = data_brain[sbj].select(rois_list[roi])        # Brain data
    x_labels = data_brain[sbj].get_label(
        label_name)  # Image labels in the brain data

    # Target features and image labels (file names)
    if feat not in loaded_features:
        y = data_features.get_features(feat)

        # select units randomlly
        y = y.reshape(y.shape[0], np.prod(y.shape[1:]))
        sample_index = np.random.choice(np.arange(np.shape[1], dtype=np.int64), size=n_sample, replace=False)
        y = y[:, sample_index]

        # save sampled index
        results_index_dir = os.path.join(results_dir_root, network, feat)
        save_array('index_random.mat', sample_index, key='index_random', dtype=np.float32, sparse=False)

        y_labels = data_features.labels

    print('Elapsed time (data preparation): %f' % (time() - start_time))

    # Calculate normalization parameters
    # ----------------------------------

    # Normalize X (fMRI data)
    # np.newaxis was added to match Matlab outputs
    x_mean = np.mean(x, axis=0)[np.newaxis, :]
    x_norm = np.std(x, axis=0, ddof=1)[np.newaxis, :]

    # Normalize Y (DNN features)
    y_mean = np.mean(y, axis=0)[np.newaxis, :]
    y_norm = np.std(y, axis=0, ddof=1)[np.newaxis, :]

    # Y index to sort Y by X (matching samples)
    # -----------------------------------------
    y_index = np.array([np.where(np.array(y_labels) == xl)
                        for xl in x_labels]).flatten()

    # Save normalization parameters
    # -----------------------------
    print('Saving normalization parameters.')
    norm_param = {'x_mean': x_mean, 'y_mean': y_mean,
                  'x_norm': x_norm, 'y_norm': y_norm}
    save_targets = [u'x_mean', u'y_mean', u'x_norm', u'y_norm']
    for sv in save_targets:
        save_file = os.path.join(results_dir, sv + '.mat')
        if not os.path.exists(save_file):
            try:
                save_array(
                    save_file, norm_param[sv], key=sv, dtype=np.float32, sparse=False)
                print('Saved %s' % save_file)
            except IOError:
                warnings.warn(
                    'Failed to save %s. Possibly double running.' % save_file)

    # Preparing learning
    # ------------------
    model = FastL2LiR()
    model_param = {'alpha':  alpha,
                   'n_feat': num_voxel[roi]}

    # Distributed computation setup
    # -----------------------------
    makedir_ifnot('./tmp')
    distcomp_db = os.path.join('./tmp', analysis_basename + '.db')
    distcomp = DistComp(backend='sqlite3', db_path=distcomp_db)

    # Model training
    # --------------
    print('Model training')
    start_time = time()

    train = ModelTraining(model, x, y)
    train.id = analysis_basename + '-' + sbj + '-' + roi + '-' + feat
    train.model_parameters = model_param

    train.X_normalize = {'mean': x_mean,
                         'std': x_norm}
    train.Y_normalize = {'mean': y_mean,
                         'std': y_norm}
    train.Y_sort = {'index': y_index}

    train.dtype = np.float32
    train.chunk_axis = chunk_axis
    train.save_format = 'bdmodel'
    train.save_path = results_dir
    train.distcomp = distcomp

    train.run()

    print('Total elapsed time (model training): %f' % (time() - start_time))
