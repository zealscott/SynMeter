import pandas as pd
import numpy as np
from lib.commons import load_json, normalize, load_config, read_csv
from lib.info import ROOT_DIR
import pickle
import os
from evaluator.privacy.util import *
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import json


def train_sample_synthesizer(model, dataset, n_gen, cuda):
    """
    Trains and samples a synthesizer model.

    Args:
        model (str): The name of the model.
        dataset (str): The name of the dataset.
        n_gen (int): The number of models to train and sample.
        cuda (bool): Whether to use CUDA for training.

    Returns:
        dict: A dictionary containing the membership information for each data point.
    """
    # load template config
    model_config = "exp/{0}/{1}/config.toml".format(dataset, model)
    config = load_config(os.path.join(ROOT_DIR, model_config))

    path_params = config["path_params"]
    # load all data
    all_data_pd, meta_data, discrete_columns = read_csv(path_params["raw_data"], path_params["meta_data"])

    dup_list = find_duplicates(all_data_pd)

    membership_info = {}
    for i in range(len(all_data_pd)):
        membership_info[i] = []

    # perpare saved dir
    privacy_dir = os.path.join(ROOT_DIR, "exp", dataset, model, "privacy")

    os.makedirs(privacy_dir, exist_ok=True)

    # dynamically import model interface
    synthesizer = __import__("evaluator.privacy." + model, fromlist=[model])

    # pack all data since they are static
    data = [all_data_pd, discrete_columns, meta_data, dup_list]
    # train model and sample
    for i in range(n_gen):
        print("start training {0}/{1}  model".format(i, n_gen))
        try:
            synthesizer.train_and_sample(config, data, membership_info, i, privacy_dir, cuda)
        except Exception as e:
            print("training {0}/{1} model failed".format(i, n_gen))
            print(e)
            continue

    # save membership info
    with open(os.path.join(privacy_dir, "membership_info.pkl"), "wb") as f:
        pickle.dump(membership_info, f)

    return membership_info


def privacy_evaluation(config, n_gen, model, dataset, cuda):
    # train and sample synthetic data
    membership_info = train_sample_synthesizer(model, dataset, n_gen, cuda)
    print("finish training and sampling, begin to evaluate privacy")

    with open(
        os.path.join(ROOT_DIR, "exp", dataset, model, "privacy", "membership_info.pkl"),
        "rb",
    ) as f:
        membership_info = pickle.load(f)

    member_nbrs_dist = {}
    for id, _ in membership_info.items():
        member_nbrs_dist[id] = []

    privacy_dir = os.path.join(ROOT_DIR, "exp", dataset, model, "privacy")

    raw_data_path = config["path_params"]["raw_data"]
    meta_data_path = config["path_params"]["meta_data"]

    # get the distance for each row in syn data
    for t in range(n_gen):
        syn_data_path = os.path.join(privacy_dir, "sampled_{}.csv".format(t))
        raw_data_arr, syn_data_arr, n_features = normalize_data(raw_data_path, syn_data_path, meta_data_path)
        distances = nearest_neighbors(syn_data_arr, raw_data_arr)
        for id, dist in enumerate(distances):
            normalized_dist = dist[0] / np.sqrt(n_features)
            member_nbrs_dist[id].append(normalized_dist)

    # compute the disclosure scores for each record when in or not in the training set
    disclosure_scores = {}
    not_in_cnt = 0
    for id, label in membership_info.items():
        print("id: {}".format(id))
        print("label: {}".format(label))
        in_label = [index for index, value in enumerate(label) if value == 1 and index < n_gen]
        out_label = [index for index, value in enumerate(label) if value == 0 and index < n_gen]
        print("id {} in_label: {}, out_label: {}".format(id, len(in_label), len(out_label)))
        assert len(in_label) + len(out_label) == n_gen
        if len(in_label) == 0 or len(out_label) == 0:
            print("id {} is not in in_label or out_label, omit".format(id))
            not_in_cnt += 1
            continue
        print("id {} in/out ratio: {}".format(id, len(in_label) / len(out_label)))
        in_dist = [member_nbrs_dist[id][i] for i in in_label]
        out_dist = [member_nbrs_dist[id][i] for i in out_label]

        # compute pairwise distance difference
        dist_diff = []
        for i in in_dist:
            for o in out_dist:
                dist_diff.append(abs(o - i))
        # average distance difference for each data point
        disclosure_scores[id] = sum(dist_diff) / len(dist_diff)

    print("not in cnt: {}".format(not_in_cnt))
    # membership disclosure score
    MDS = max(disclosure_scores.values())
    print("membership disclosure score: {}".format(MDS))
    # save the distance difference
    with open(os.path.join(privacy_dir, "disclosure_score.pkl"), "wb") as f:
        pickle.dump(disclosure_scores, f)

    return {"MDS": MDS}


def save_privacy_results(res, path):
    """
    Save the privacy evaluation results to a json file.

    Args:
        res (dict): The privacy evaluation results.
        path (str): The path to save the results.
    """
    with open(path, "w") as f:
        json.dump(res, f)

    print("Privacy evaluation results saved to {0}".format(path))
