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


def train_sample_synthesizer(model, dataset, m_shadow_models, n_syn_dataset, cuda):
    """
    Trains and samples a synthesizer model.

    Args:
        model (str): The name of the model.
        dataset (str): The name of the dataset.
        m_shadow_models (int): The number of synthesizer to train.
        n_syn_dataset (int): The number of synthetic datasets to generate.
        cuda (bool): Whether to use CUDA for training.

    Returns:
        dict: A dictionary containing the membership information for each data point.
    """
    # perpare saved dir
    privacy_dir = os.path.join(ROOT_DIR, "exp", dataset, model, "privacy")
    os.makedirs(privacy_dir, exist_ok=True)

    # load template config
    model_config = "exp/{0}/{1}/config.toml".format(dataset, model)
    config = load_config(os.path.join(ROOT_DIR, model_config))
    path_params = config["path_params"]
    # load all data
    all_data_pd, meta_data, discrete_columns = read_csv(path_params["raw_data"], path_params["meta_data"])

    # duplicate
    clean_data_pd = all_data_pd.drop_duplicates().reset_index(drop=True)

    # we use the same technique as shadow model training
    # make sure each sample is seen exactly once
    size = len(clean_data_pd)
    np.random.seed(0)
    keep = np.random.uniform(0, 1, size=(m_shadow_models, size))
    order = keep.argsort(0)
    keep = order < int(0.5 * m_shadow_models)

    if "privsyn" in model:
        model_name = "privsyn"
    elif "mst" in model:
        model_name = "mst"
    elif "tablediffusion" in model:
        model_name = "tablediffusion"
    elif "pategan" in model:
        model_name = "pategan"
    elif "tabsyn" in model:
        model_name = "tabsyn"
    else:
        model_name = model

    # dynamically import model interface
    synthesizer = __import__("evaluator.privacy." + model_name, fromlist=[model_name])

    # train model and sample
    for shadow_id in range(m_shadow_models):
        print("start training {0}/{1}  model".format(shadow_id, m_shadow_models))
        # perpare saved dir
        cur_shadow_dir = os.path.join(privacy_dir, str(shadow_id))
        os.makedirs(cur_shadow_dir, exist_ok=True)

        # membership info
        cur_keep = np.array(keep[shadow_id], dtype=bool)
        cur_member = cur_keep.nonzero()[0]

        # select data for this shadow model
        shadow_data_pd = clean_data_pd.iloc[cur_member].reset_index(drop=True)
        print(f"{shadow_id} shadow data size: {len(shadow_data_pd)}/{len(clean_data_pd)}")

        # pack all data
        data = [shadow_data_pd, discrete_columns, meta_data]

        synthesizer.train_and_sample(config, data, cur_shadow_dir, cuda, n_syn_dataset)

        # save membership info
        with open(os.path.join(cur_shadow_dir, "member.pkl"), "wb") as f:
            pickle.dump(cur_member, f)


def compute_MDS(model, dataset, m_shadow_models, n_syn_dataset):
    privacy_dir = os.path.join(ROOT_DIR, "exp", dataset, model, "privacy")

    # load template config
    model_config = "exp/{0}/{1}/config.toml".format(dataset, model)
    config = load_config(os.path.join(ROOT_DIR, model_config))
    path_params = config["path_params"]
    # load all data
    all_data_pd, meta_data, discrete_columns = read_csv(path_params["raw_data"], path_params["meta_data"])

    # duplicate
    print(f"original data size: {len(all_data_pd)}")
    clean_data_pd = all_data_pd.drop_duplicates().reset_index(drop=True)
    print(f"duplicate data size: {len(clean_data_pd)}")

    # init the distance dict
    in_member_nbrs_dist = {}
    out_member_nbrs_dist = {}
    for id in range(len(clean_data_pd)):
        in_member_nbrs_dist[id] = []
        out_member_nbrs_dist[id] = []

    for shadow_id in range(m_shadow_models):
        cur_shadow_dir = os.path.join(privacy_dir, str(shadow_id))
        cur_shadow_dist = {}
        for id in range(n_syn_dataset):
            syn_data_path = os.path.join(cur_shadow_dir, "sampled_{}.csv".format(id))
            raw_data_arr, syn_data_arr, n_features = normalize_data(
                clean_data_pd, syn_data_path, path_params["meta_data"]
            )
            distances = nearest_neighbors(syn_data_arr, raw_data_arr)

            for i, dist in enumerate(distances):
                normalized_dist = dist[0] / np.sqrt(n_features)
                if i not in cur_shadow_dist:
                    cur_shadow_dist[i] = [normalized_dist]
                else:
                    cur_shadow_dist[i].append(normalized_dist)
        with open(os.path.join(cur_shadow_dir, "member.pkl"), "rb") as f:
            member = pickle.load(f)
        # get the expected distance for each record
        for id, dist_list in cur_shadow_dist.items():
            mean_dist = np.mean(dist_list)
            if id in member:
                in_member_nbrs_dist[id].append(mean_dist)
            else:
                out_member_nbrs_dist[id].append(mean_dist)

    # get the DS for each record
    DS = {}
    for id in range(len(clean_data_pd)):
        mean_in_dist = np.mean(in_member_nbrs_dist[id])
        mean_out_dist = np.mean(out_member_nbrs_dist[id])
        DS[id] = abs(mean_in_dist - mean_out_dist)
        print(
            f"for record {id}, in member size: {len(in_member_nbrs_dist[id])}, out member size: {len(out_member_nbrs_dist[id])}, DS: {DS[id]}"
        )

    # get the MDS
    MDS = max(DS.values())

    print("membership disclosure score: {}".format(MDS))

    plt.style.use("science")
    sns.set_theme()
    sns.set_style("whitegrid")
    sns.set_palette("Set2")
    plt.figure(figsize=(8, 6))
    plt.hist(list(DS.values()), bins=50)
    plt.xlabel("distance difference")
    plt.ylabel("frequency")
    plt.title("disclosure score distribution")
    plt.savefig(os.path.join(privacy_dir, "disclosure_score.png"))

    # save the distance difference
    with open(os.path.join(privacy_dir, "disclosure_score.pkl"), "wb") as f:
        pickle.dump(DS, f)

    return MDS


def privacy_evaluation(config, m_shadow_models, n_syn_dataset, model, dataset, cuda):
    # train and sample synthetic data
    train_sample_synthesizer(model, dataset, m_shadow_models, n_syn_dataset, cuda)
    print("finish training and sampling, begin to evaluate privacy")

    # compute the membership disclosure score
    MDS = compute_MDS(model, dataset, m_shadow_models, n_syn_dataset)

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
