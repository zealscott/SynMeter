import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from lib.commons import load_json, normalize, cat_encode
import sklearn.preprocessing


def nearest_neighbors(syn_data: np.array, target_data: np.array):
    """
    Find the nearest distance in syn_data for each target data.

    Parameters:
        syn_data (np.array): The synthetic data array.
        target_data (np.array): The target data array.

    Returns:
        np.array: An array of distances representing the nearest distance in syn_data for each target data.
    """
    nbrs_synth = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=2).fit(syn_data)
    distances, _ = nbrs_synth.kneighbors(target_data)
    return distances


def normalize_data(raw_data_pd, syn_data_path, meta_data_path):
    """
    Load raw data, meta data, and synthetic data.
    Encode all data to numerical data and normalize the data to the range [0, 1].

    Parameters:
        raw_data_path (pd.DataFrame): The raw data.
        syn_data_path (str): The file path of the synthetic data.
        meta_data_path (str): The file path of the meta data.

    Returns:
        raw_data_arr (numpy.ndarray): The normalized raw data.
        syn_data_arr (numpy.ndarray): The normalized synthetic data.
    """
    syn_data_pd = pd.read_csv(syn_data_path)
    meta_data = load_json(meta_data_path)
    discrete_cols = [col["name"] for col in meta_data["columns"] if col["type"] != "continuous"]
    normalization = "minmax"

    # remove nan or inf values in the synthetic data
    syn_data_pd = syn_data_pd.replace([np.inf, -np.inf], np.nan)
    syn_data_pd = syn_data_pd.dropna()

    raw_data_arr = []
    syn_data_arr = []

    for col in raw_data_pd.columns:
        real_data_col = raw_data_pd[col].values.reshape(-1, 1)
        syn_data_col = syn_data_pd[col].values.reshape(-1, 1)
        # set the type of synthetic data to be the same as the real data
        syn_data_col = syn_data_col.astype(real_data_col.dtype)
        # fit the scaler
        if col in discrete_cols:
            scaler = cat_encode(real_data_col)
        else:
            scaler = normalize(real_data_col, normalization)

        # transform the data
        real_data_transformed = scaler.transform(real_data_col)
        syn_data_transformed = scaler.transform(syn_data_col)

        # ensure the transformed data has 2 dimensions
        if len(real_data_transformed.shape) == 1:
            real_data_transformed = real_data_transformed.reshape(-1, 1)
        if len(syn_data_transformed.shape) == 1:
            syn_data_transformed = syn_data_transformed.reshape(-1, 1)

        raw_data_arr.append(real_data_transformed)
        syn_data_arr.append(syn_data_transformed)

    # concatenate the features in the last dimension
    raw_data_arr = np.concatenate(raw_data_arr, axis=1)
    syn_data_arr = np.concatenate(syn_data_arr, axis=1)

    n_features = len(raw_data_pd.columns)

    return raw_data_arr, syn_data_arr, n_features


def sample_half_data(all_data_pd, dup_list, membership_info):
    """
    Sample half of the data from all data, exclude duplicates, and save in membership_info.

    Parameters:
        all_data_pd (pandas.DataFrame): The dataframe containing all the data.
        dup_list (list): A list of lists, where each inner list represents a group of duplicate data.
        membership_info (dict): A dictionary where the keys are IDs and the values are lists representing membership information.

    Returns:
        tuple: A tuple containing the train_index (list)
    """
    # randomly use half of the data to train the model, sample index and to list
    train_index = np.random.choice(len(all_data_pd), int(len(all_data_pd) / 2), replace=False)
    train_index = list(train_index)
    drop_index = list(set(range(len(all_data_pd))) - set(train_index))
    # when data in train, make sure that the duplicate data is also in train
    for cur_dup_list in dup_list:
        if len(set(cur_dup_list) - set(drop_index)) != 0:
            # if some duplicate data is not in train, all duplicate data should not be in train
            # union the duplicate data and drop_index
            drop_index = list(set(drop_index) | set(cur_dup_list))
            # remove duplicate data from train
            train_index = list(set(train_index) - set(cur_dup_list))

    print("after deduplication, train size ratio: ", len(train_index) / len(all_data_pd))

    # append the membership info {id:[0,1,0,...]}
    for cur_id, cur_membership_info in membership_info.items():
        if cur_id in train_index:
            cur_membership_info.append(1)
        else:
            cur_membership_info.append(0)

    return train_index
