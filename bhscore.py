import os
import argparse

import numpy as np
import json
from scipy.stats import spearmanr, t
import hdf5storage
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--json', '-j', default='bhscore_dnn_options.json', help='json file that contains options')
parser.add_argument('--output', '-o', default=None, help='output file path')
parser.add_argument('--nunit', type=int, default=1000, help='Number of unit to select from each layer')
parser.add_argument('--pval', '-p', type=float, default=0.05,
                    help='threshold of p value for unit selection based on prediction accuracy')
parser.add_argument('--load', action='store_true')
opt = parser.parse_args()


def load_data(results_options, subjects, rois, nunit):
    """
    results_options: see examples
    subjects: list of subject names
    rois: list of roi names

    return : dictionary that contains prediction accuracies of selected DNN units in each layer from each ROI 
    """
    ret = dict()
    for dnn in tqdm(results_options.keys()):
        layers = results_options[dnn]['layers']
        acc_path = results_options[dnn]['dir']

        predacc_alllayer = []
        for layer in layers:

            predacc_allsub = []
            for sub in subjects:

                predacc_allroi = []
                for roi in rois:
                    # Load predicted features
                    d = hdf5storage.loadmat(os.path.join(acc_path, layer, sub, roi, 'accuracy.mat'))
                    predacc = d['accuracy'].flatten()

                    # select specified number of units
                    if nunit < len(predacc):
                        np.random.seed(1)
                        predacc = np.random.choice(predacc, size=nunit, replace=False)

                    predacc_allroi.append(predacc)

                predacc_allroi = np.stack(predacc_allroi)
                predacc_allsub.append(predacc_allroi)

            predacc_allsub = np.hstack(predacc_allsub)
            predacc_alllayer.append(predacc_allsub)

        ret[dnn] = predacc_alllayer

    return ret


def compute_bhscore(predacc_list, pval_threshold):
    """
    predacc_list: sequence of ndarray. Each ndarray represents prediction accuracies of DNN units from each ROI.  
    Length of predacc_list should be the same as the number of layers. 
    The shape of each ndarray should be [# of units x # of ROIs]. 
    """

    best_rois = []
    for predacc in predacc_list:
        # if prediction accuracy is nan, convert it to zero
        predacc[np.isnan(predacc)] = 0

        # if a unit cannot be predicted from all ROIs, remove it
        for i in range(predacc.shape[0] - 1, 0, -1):
            if np.sum(predacc[:, i]) == 0:
                predacc = np.delete(predacc, i, 1)

        # for each CNN units, search roi which has the highest prediction accuracy
        pred_max = np.max(predacc, axis=0)
        pred_max_ind = np.argmax(predacc, axis=0)

        # compute p value of the highest decoding accuracy
        tmp = np.sqrt((50 - 2) * (1 - pred_max ** 2))
        tmp = pred_max * tmp
        pval = 2 * (1 - t.cdf(tmp, df=50 - 2))

        # keep unit with p value < threshold and acc > 0
        threshold = pval < opt.pval
        plus_unit = pred_max > 0
        select_unit_ind = np.logical_and(threshold, plus_unit)
        pred_max_ind = pred_max_ind[select_unit_ind]

        best_rois.append(pred_max_ind)

    # get layer numbers of each unit. concatenate best ROIs for all layers
    layer_numbers = []
    best_roi_flatten = []
    for i_br, br in enumerate(best_rois):
        layer_numbers.extend(np.repeat(i_br + 1, len(br)))
        best_roi_flatten.extend(br)

    # compute Spearman's rank correlation
    bhscore, _ = spearmanr(layer_numbers, best_roi_flatten)

    return bhscore, best_rois


def random_sample_bhscore(predacc_list, pval_threshold, n_sample_layer=5, n_sample=100):
    """
    Randomly sample specified number of layers and compute mean BH score
    """

    bhscore_list = np.zeros(n_sample)
    for i_s in range(n_sample):
        # sample layers
        sample_index = np.random.choice(np.arange(1, len(predacc_list)-1), size=n_sample_layer - 2, replace=False)
        sample_index = np.sort(sample_index)
        predacc_list_sampled = [predacc_list[0]] + [predacc_list[i] for i in sample_index] + [predacc_list[-1]]

        bhscore, _ = compute_bhscore(predacc_list_sampled, pval_threshold)
        bhscore_list[i_s] = bhscore

    return bhscore_list


def main():
    # Load settings
    with open(opt.json, 'r') as f:
        dat = json.load(f)

    subjects = dat['subjects']
    rois = dat['rois']
    results_options = dat['dnns']

    # load results of feature prediction analysis
    print('Loading results of feature prediction analysis')
    results_feature_prediction = load_data(results_options, subjects, rois, opt.nunit)

    # compute BH score
    networks = []
    bhscores = []
    print('Computing BH score')
    for dnn in results_feature_prediction.keys():
        res = results_feature_prediction[dnn]
        bhscore = random_sample_bhscore(res, opt.pval)

        networks.append(dnn)
        bhscores.append(bhscore)

        print('%s: %.2f' % (dnn, np.mean(bhscore)))

    # write result to file
    if opt.output is None:
        output_path = __file__.replace('.py', '.mat')
    else:
        output_path = opt.output
    mdict = {'bhscores': bhscores,
             'networks': networks}
    hdf5storage.savemat(output_path, mdict, trancate_existing=True)


if __name__ == '__main__':
    main()
