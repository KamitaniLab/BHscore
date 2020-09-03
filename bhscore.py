'''A Module to compute the BH score.

Author: Souma Nonaka, Shuntaro C. Aoki
'''


import os

import numpy as np
from scipy.stats import spearmanr, t


def compute_bhscore(predacc_list, pval=0.05, return_top_rois=False):
    """Compute a BH score of a given DNN.

    Parameters
    ----------
    predacc_list : list of arrays
        List of prediction accuracies for a DNN. Each array contains
        prediction accuracies of individual units in a layer, formed as an
         array of ROIs x units.
    pval : float, default = 0.05
        P-value threshold in unit selection.
    return_top_rois : bool, default = False
        Returns top ROIs if True.

    Returns
    -------
    bhscore : float
    top_rois: list of arrays
    """

    top_rois = []
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

        top_rois.append(pred_max_ind)

    # get layer numbers of each unit. concatenate best ROIs for all layers
    layer_numbers = []
    best_roi_flatten = []
    for i_br, br in enumerate(top_rois):
        layer_numbers.extend(np.repeat(i_br + 1, len(br)))
        best_roi_flatten.extend(br)

    # compute Spearman's rank correlation
    bhscore, _ = spearmanr(layer_numbers, best_roi_flatten)

    if return_top_rois:
        return bhscore, top_rois
    else:
        return bhscore


def compute_bhscore_layerselect(predacc_list, pval=0.05, n_layers=5,
                                n_repeat=100, return_top_rois=False):
    """Compute a BH score of a given DNN, random layer selection version.

    Parameters
    ----------
    predacc_list : list of arrays
        List of prediction accuracies for a DNN. Each array contains
        prediction accuracies of individual units in a layer, formed as an
         array of ROIs x units.
    pval : float, default = 0.05
        P-value threshold in unit selection.
    n_layers : int, default = 5
        The number of layers used to compute the BH score. Note that the first
        and last layers are always included in the computation. Thus,
        (n_layers - 2) layers are randomly selected from the representative
        layers except the first and last ones.
    n_repeat : int, default = 100
        The number of random layer selection.
    return_top_rois : bool, default = False
        Returns top ROIs if True.

    Returns
    -------
    bhscore_list : arary of float
    top_rois_list : list of list of arrays
    """

    bhscore_list = np.zeros(n_repeat)
    top_rois_list = []
    for i_s in range(n_repeat):
        # sample layers
        sample_index = np.random.choice(np.arange(1, len(predacc_list)-1), size=n_layers - 2, replace=False)
        sample_index = np.sort(sample_index)
        predacc_list_sampled = [predacc_list[0]] + [predacc_list[i] for i in sample_index] + [predacc_list[-1]]

        bhscore, top_rois = compute_bhscore(predacc_list_sampled, pval, return_top_rois=True)
        bhscore_list[i_s] = bhscore
        top_rois_list.append(top_rois)

    if return_top_rois:
        return bhscore_list, top_rois_list
    else:
        return bhscore_list


if __name__ == '__main__':
    pass
