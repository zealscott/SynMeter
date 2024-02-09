"""
this file contains functions for column-wise and pairwise waterstein distance
"""
import pandas as pd
import numpy as np
from lib.commons import normalize
from scipy.stats import wasserstein_distance
from evaluator.utility.query import cal_all_cond_prob
import torch
import ot


def cal_fidelity(real_data, syn_data, dis_col_value_dict, normalization="minmax"):
    # first, compute column-wise distance
    ret = {}

    print("computing cat_error")
    _cat_error = cat_error(real_data, syn_data, dis_col_value_dict)
    if _cat_error:
        ret["cat_error"] = _cat_error

    print("computing num_error")
    _num_error = num_error(real_data, syn_data, dis_col_value_dict)
    if _num_error:
        ret["cont_error"] = _num_error

    print("computing num_num_error")
    _num_num_error = num_num_error(real_data, syn_data, dis_col_value_dict)
    if _num_num_error:
        ret["cont_cont_error"] = _num_num_error

    print("computing cat_num_error")
    _cat_num_error = cat_num_error(real_data, syn_data, dis_col_value_dict)
    if _cat_num_error:
        ret["cat_cont_error"] = _cat_num_error

    print("computing cat_cat_error")
    _cat_cat_error = cat_cat_error(real_data, syn_data, dis_col_value_dict)
    if _cat_cat_error:
        ret["cat_cat_error"] = _cat_cat_error

    return ret


###########################################################################
#########################2-way Wasserstein distance########################
###########################################################################


def cat_num_error(real_data, syn_data, dis_col_value_dict):
    """
    compute the cat-num error (2-way) for discrete and continuous columns between real and syn data
    """
    discrete_cols = list(dis_col_value_dict.keys())
    num_cols = [col for col in real_data.columns if col not in discrete_cols]

    # for each combination of discrete and continuous columns, compute the 2D wasserstein distance
    wasserstein_error = []
    for i in range(len(discrete_cols)):
        for j in range(len(num_cols)):
            cat_col, num_col = discrete_cols[i], num_cols[j]
            real_cat = real_data[cat_col].values.reshape(-1, 1)
            syn_cat = syn_data[cat_col].values.reshape(-1, 1)
            real_num = real_data[num_col].values.reshape(-1, 1)
            syn_num = syn_data[num_col].values.reshape(-1, 1)
            # compute the categorical distance matrix
            cat_dist_matrix = np.not_equal(real_cat[:, None], syn_cat).astype(int).squeeze()
            # normalize the numerical column to [0, 1]
            scaler = normalize(np.concatenate([real_num, syn_num]), normalization="minmax")
            norm_real_num = scaler.transform(real_num)
            norm_syn_num = scaler.transform(syn_num)
            # compute the numerical distance matrix
            num_dist_matrix = ot.dist(norm_real_num, norm_syn_num, metric="minkowski", p=1)
            # compute the 2D wasserstein distance with linear programming
            # cost = 1(cat_real == cat_syn) + |num_real - num_syn|
            cost_matrix = cat_dist_matrix + num_dist_matrix
            wasserstein_error.append(ot.emd2([], [], cost_matrix))  # no need to assign weights

    return np.nanmean(wasserstein_error) if wasserstein_error else None


def num_num_error(real_data, syn_data, dis_col_value_dict):
    """
    compute the numerical error (2-way) for numerical columns between real and syn data
    """
    discrete_cols = list(dis_col_value_dict.keys())
    continous_cols = [col for col in real_data.columns if col not in discrete_cols]

    # for each combination of continuous columns, compute the 2D wasserstein distance
    wasserstein_error = []
    for i in range(len(continous_cols)):
        for j in range(i + 1, len(continous_cols)):
            col1, col2 = continous_cols[i], continous_cols[j]
            real_col1 = real_data[col1].values.reshape(-1, 1)
            real_col2 = real_data[col2].values.reshape(-1, 1)
            syn_col1 = syn_data[col1].values.reshape(-1, 1)
            syn_col2 = syn_data[col2].values.reshape(-1, 1)
            # normalize the column to [0, 1]
            scaler1 = normalize(np.concatenate([real_col1, syn_col1]), normalization="minmax")
            scaler2 = normalize(np.concatenate([real_col2, syn_col2]), normalization="minmax")
            norm_real_col1 = scaler1.transform(real_col1).flatten()
            norm_syn_col1 = scaler1.transform(syn_col1).flatten()
            norm_real_col2 = scaler2.transform(real_col2).flatten()
            norm_syn_col2 = scaler2.transform(syn_col2).flatten()
            # compute the 2D wasserstein distance with linear programming
            # concatenate the two columns
            real_col = np.concatenate([norm_real_col1.reshape(-1, 1), norm_real_col2.reshape(-1, 1)], axis=1)
            syn_col = np.concatenate([norm_syn_col1.reshape(-1, 1), norm_syn_col2.reshape(-1, 1)], axis=1)
            # use 1-norm as the distance metric
            cost_matrix = ot.dist(real_col, syn_col, metric="minkowski", p=1)
            wasserstein_error.append(ot.emd2([], [], cost_matrix))  # no need to assign weights
    return np.nanmean(wasserstein_error) if wasserstein_error else None


def cat_cat_error(real_data, syn_data, dis_col_value_dict):
    """
    compute the contigency error (2-way) for discrete columns between real and syn data
    """
    discrete_cols = list(dis_col_value_dict.keys())

    contigency_error = []
    for i in range(len(discrete_cols)):
        for j in range(i + 1, len(discrete_cols)):
            col1, col2 = discrete_cols[i], discrete_cols[j]
            val1, val2 = dis_col_value_dict[col1], dis_col_value_dict[col2]
            marginal_diff = marginal_query(real_data, syn_data, [col1, col2], [val1, val2])
            contigency_error.append(marginal_diff * 0.5)
    return np.nanmean(contigency_error) if contigency_error else None


###########################################################################
#########################1-way Wasserstein distance########################
###########################################################################


def num_error(real_data, syn_data, dis_col_value_dict):
    """
    compute the categorical error (1-way) for numerical columns between real and syn data
    """
    discrete_cols = list(dis_col_value_dict.keys())
    wasserstein_error = []
    for column in real_data.columns:
        if column not in discrete_cols:
            real_col = real_data[column].values.reshape(-1, 1)
            syn_col = syn_data[column].values.reshape(-1, 1)
            # normalize the column to [0, 1]
            scaler = normalize(np.concatenate([real_col, syn_col]), normalization="minmax")
            norm_real_col = scaler.transform(real_col).flatten()
            norm_syn_col = scaler.transform(syn_col).flatten()
            wasserstein_error.append(wasserstein_distance(norm_real_col, norm_syn_col))

    return np.nanmean(wasserstein_error) if wasserstein_error else None


def cat_error(real_data, syn_data, dis_col_value_dict):
    """
    compute the marginal error (1-way) for discrete columns between real and syn data
    """
    discrete_cols = list(dis_col_value_dict.keys())

    marginal_error = []
    for i in range(len(discrete_cols)):
        col = discrete_cols[i]
        val = dis_col_value_dict[col]
        marginal_diff = marginal_query(real_data, syn_data, [col], [val])
        marginal_error.append(marginal_diff * 0.5)
    return np.nanmean(marginal_error) if marginal_error else None


def marginal_query(real_data, syn_data, cols, values):
    """
    real_probs: list of marginal probabilities for real data
    syn_probs: list of marginal probabilities for syn data
    calulate the average absolute difference between real_probs and syn_probs
    """
    real_probs = cal_all_cond_prob(real_data, cols, values)
    syn_probs = cal_all_cond_prob(syn_data, cols, values)
    try:
        assert sum(real_probs) >= 1 - 1e-2
        assert sum(syn_probs) >= 1 - 1e-2
    except:
        print("error in marginal_query for cols: ", cols)
        print("real_probs: ", sum(real_probs))
        print("syn_probs: ", sum(syn_probs))
        raise ValueError("sum of probs should be 1")
    abs_diff = np.abs(np.array(real_probs) - np.array(syn_probs))
    return sum(abs_diff)
