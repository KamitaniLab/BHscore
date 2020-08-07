import os

import numpy as np
from scipy.stats import spearmanr, t


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

        # for each CNN units, search roi which has the highest prediction accuracy
        pred_max = np.max(predacc, axis=0)
        pred_max_ind = np.argmax(predacc, axis=0)

        # compute p value of the highest decoding accuracy
        tmp = np.sqrt((50 - 2) * (1 - pred_max ** 2))
        tmp = pred_max * tmp
        pval = 2 * (1 - t.cdf(tmp, df=50 - 2))

        # keep unit with p value < threshold and acc > 0
        threshold = pval < 0.05
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


if __name__ == '__main__':
    pass
