'''DNN Feature encoding - encoders training script'''


from __future__ import print_function

from itertools import product
import os
from time import time
import warnings
import argparse

import bdpy
from bdpy.dataform import Features, save_array
from bdpy.distcomp import DistComp
from bdpy.ml import ModelTraining
from bdpy.util import makedir_ifnot
from fastl2lir import FastL2LiR
import numpy as np
import yaml


# Main #######################################################################

def main(conf):

    # Settings ###############################################################

    # Brain data
    subjects_list = conf['training fmri']
    label_key = conf['label key']
    rois_list = conf['rois']
    num_unit = conf['layers unit num']

    # Image features
    features_dir = conf['training feature dir'][0]
    network = conf['network']
    features_list = conf['layers']
    features_list = features_list[::-1]  # Start training from deep layers

    if 'feature index file' in conf:
        feature_index_file = os.path.join(features_dir, network, conf['feature index file'])
    else:
        feature_index_file = None

    # Model parameters
    alpha = conf['encoding alpha']

    # Results directory
    if 'analysis_name' in conf:
        results_dir_root = os.path.join(conf['feature encoder dir'], conf['analysis name'])
    else:
        results_dir_root = conf['feature encoder dir']


    # Main ###################################################################

    analysis_basename = os.path.splitext(os.path.basename(__file__))[0] + '-' + conf['__filename__']

    # Print info -------------------------------------------------------------
    print('Subjects:        %s' % list(subjects_list.keys()))
    print('ROIs:            %s' % list(rois_list.keys()))
    print('DNN features:    %s' % network)
    print('Layers:          %s' % features_list)
    print('')

    # Load data --------------------------------------------------------------
    print('----------------------------------------')
    print('Loading data')

    data_brain = {sbj: bdpy.BData(dat_file[0])
                  for sbj, dat_file in subjects_list.items()}

    if feature_index_file is not None:
        data_features = Features(os.path.join(features_dir, network), feature_index=feature_index_file)
    else:
        data_features = Features(os.path.join(features_dir, network))

    # Initialize directories -------------------------------------------------
    makedir_ifnot(results_dir_root)
    makedir_ifnot(os.path.join(results_dir_root, network))
    makedir_ifnot('tmp')

    # Analysis loop ----------------------------------------------------------
    print('----------------------------------------')
    print('Analysis loop')

    for feat, sbj, roi in product(features_list, subjects_list, rois_list):
        print('--------------------')
        print('Feature:    %s' % feat)
        print('Subject:    %s' % sbj)
        print('ROI:        %s' % roi)
        print('Num unit:   %d' % num_unit[feat])

        # Setup
        # -----
        analysis_id = analysis_basename + '-' + sbj + '-' + roi + '-' + feat
        results_dir = os.path.join(results_dir_root, network, feat, sbj, roi, 'model')
        makedir_ifnot(results_dir)

        # Check whether the analysis has been done or not.
        info_file = os.path.join(results_dir, 'info.yaml')
        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                info = yaml.load(f)
            while info is None:
                warnings.warn('Failed to load info from %s. Retrying...'
                              % info_file)
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

        # DNN features and image labels (file names)
        x = data_features.get_features(feat)  # DNN features
        x = x.astype(np.float32)
        x = x.reshape((x.shape[0], np.prod(x.shape[1:])), order='F')
        x_labels = data_features.labels       # Labels

        # Brain data
        y = data_brain[sbj].select(rois_list[roi])       # Brain data
        y = y.astype(np.float32)
        y_labels = data_brain[sbj].get_label(label_key)  # Labels

        print('Elapsed time (data preparation): %f' % (time() - start_time))

        # Save feature index
        # ------------------
        if feature_index_file is not None:
            feature_index_save_file = os.path.join(results_dir, 'feature_index.mat')
            data_features.save_feature_index(feature_index_save_file)
            print('Saved %s' % feature_index_file)

        # Calculate normalization parameters
        # ----------------------------------

        # Normalize X (fMRI data)
        x_mean = np.mean(x, axis=0)[np.newaxis, :]  # np.newaxis was added to match Matlab outputs
        x_norm = np.std(x, axis=0, ddof=1)[np.newaxis, :]

        # Normalize Y (DNN features)
        y_mean = np.mean(y, axis=0)[np.newaxis, :]
        y_norm = np.std(y, axis=0, ddof=1)[np.newaxis, :]

        # Y index to sort Y by X (matching samples)
        # -----------------------------------------
        x_index = np.array([np.where(np.array(x_labels) == yl) for yl in y_labels]).flatten()

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
                    save_array(save_file, norm_param[sv], key=sv, dtype=np.float32, sparse=False)
                    print('Saved %s' % save_file)
                except Exception:
                    warnings.warn('Failed to save %s. Possibly double running.' % save_file)

        # Preparing learning
        # ------------------
        model = FastL2LiR()
        model_param = {'alpha':  alpha,
                       'n_feat': num_unit[feat],
                       'dtype': np.float32}

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
        train.X_sort = {'index': x_index}

        train.dtype = np.float32
        train.save_format = 'bdmodel'
        train.save_path = results_dir
        train.distcomp = distcomp

        train.run()

        print('Total elapsed time (model training): %f' % (time() - start_time))

    print('%s finished.' % analysis_basename)


# Entry point ################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'conf',
        type=str,
        help='analysis configuration file',
    )
    args = parser.parse_args()

    conf_file = args.conf

    with open(conf_file, 'r') as f:
        conf = yaml.safe_load(f)

    conf.update({
        '__filename__': os.path.splitext(os.path.basename(conf_file))[0]
    })

    main(conf)
