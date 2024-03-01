import tomli
from typing import Any, Union
from pathlib import Path
import json
import tomli_w
import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, r2_score, roc_auc_score, mean_squared_error
import torch
import torch.nn as nn
import random



# --------------------------------------------------------------- #
# ------------------- tools for process data ------------------ #
# --------------------------------------------------------------- #
def cat_encode(X):
    """
    one-hot encode for categorical and ordinal features
    """
    oe = sklearn.preprocessing.OneHotEncoder(
        handle_unknown="ignore",  # type: ignore[code]
        sparse_output=False,  # type: ignore[code]
    ).fit(X)

    return oe


def normalize(X, normalization="quantile"):
    """
    normalize continuous features
    """
    if normalization == "standard":
        scaler = sklearn.preprocessing.StandardScaler()
    elif normalization == "minmax":
        scaler = sklearn.preprocessing.MinMaxScaler()
    elif normalization == "quantile":
        # adopt from Tab-DDPM
        scaler = sklearn.preprocessing.QuantileTransformer(
            output_distribution="normal", n_quantiles=max(min(X.shape[0] // 30, 1000), 10), subsample=int(1e8)
        )
    else:
        raise ValueError("normalization must be standard, minmax, or quantile, but got " + normalization)

    scaler.fit(X)
    return scaler


def preprocess(train_data, val_data, meta_data, discrete_cols, normalization="quantile"):
    """
    convert dataframe to numpy array with one-hot encoding and normalization
    return converted numpy array and encodings
    note: we only use train and val data to fit the encodings
    """
    # combine train and val data
    data = pd.concat([train_data, val_data], ignore_index=True)

    # travse the dataframe and convert categorical and ordinal features to one-hot encoding
    encodings = {}

    # first, get the encodings for train/val data
    for col in data.columns:
        if col == "label":
            # flatten the label column
            y_arr = np.concatenate(data[col].values.flatten().reshape(-1, 1)).ravel()
            if meta_data["task"] != "regression":
                le = sklearn.preprocessing.LabelEncoder()
                le.fit(y_arr.ravel())
                encodings[col] = le
            else:
                scaler = normalize(y_arr.reshape(-1, 1), normalization)
                encodings[col] = scaler
        else:
            if col in discrete_cols:
                oe = cat_encode(data[col].values.reshape(-1, 1))
                encodings[col] = oe
            else:
                scaler = normalize(data[col].values.reshape(-1, 1), normalization)
                encodings[col] = scaler
    train_x = []
    val_x = []
    train_y = []
    val_y = []

    # then, convert train and val data
    for col in data.columns:
        if col == "label":
            # flatten the label column
            train_arr = train_data[col].values.flatten().reshape(-1, 1)
            val_arr = val_data[col].values.flatten().reshape(-1, 1)
            if meta_data["task"] != "regression":
                train_arr,val_arr = train_arr.ravel(),val_arr.ravel()
            train_y = encodings[col].transform(train_arr)
            val_y = encodings[col].transform(val_arr)
        else:
            train_temp = encodings[col].transform(train_data[col].values.reshape(-1, 1))
            val_temp = encodings[col].transform(val_data[col].values.reshape(-1, 1))
            train_x.append(train_temp)
            val_x.append(val_temp)

    train_x = np.concatenate(train_x, axis=1)
    val_x = np.concatenate(val_x, axis=1)
    return [train_x, train_y], [val_x, val_y], encodings


def transform_data(data, encodings, meta_data):
    """
    transform data with given encodings
    """
    x = []
    y = []
    for col in data.columns:
        if col == "label":
            # flatten the label column
            y = encodings[col].transform(data[col].values.reshape(-1, 1))
        else:
            x_temp = encodings[col].transform(data[col].values.reshape(-1, 1))
            x.append(x_temp)

    x = np.concatenate(x, axis=1)
    return [x, y]


# --------------------------------------------------------------- #
# ------------------------------- helper ------------------------ #
# --------------------------------------------------------------- #
def read_csv(csv_filename, meta_filename=None):
    """Read a csv file."""
    with open(meta_filename) as meta_file:
        meta_data = json.load(meta_file)

    discrete_cols = [column["name"] for column in meta_data["columns"] if column["type"] != "continuous"]

    data = pd.read_csv(csv_filename, header="infer")
    # set the discrete columns to string
    for col in discrete_cols:
        data[col] = data[col].astype(str)

    return data, meta_data, discrete_cols


def get_n_class(meta_filename):
    """
    Get the number of classes in the dataset.
    if regression, return -1
    """
    with open(meta_filename) as meta_file:
        meta_data = json.load(meta_file)

    for column in meta_data["columns"]:
        if column["name"] == "label" and column["type"] != "continuous":
            return column["size"]

    return -1


def load_config(path: Union[Path, str]) -> Any:
    """
    load config file `.toml`
    """
    with open(path, "rb") as f:
        return tomli.load(f)


def dump_config(config, path):
    """
    dump config file `.toml`
    """
    with open(path, "wb") as f:
        tomli_w.dump(config, f)
    print("successfully dump config to {0}".format(path))


def improve_reproducibility(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True


def load_json(path):
    """
    Load meta data `.json`
    """
    with open(path) as f:
        meta_data = json.load(f)
    return meta_data


# --------------------------------------------------------------- #
# --------------------- tools for ML evaluator ------------------ #
# --------------------------------------------------------------- #
def f1(net, X, y):
    y_pred = net.predict(X)
    return f1_score(y, y_pred, average="weighted")


def r2(net, X, y):
    y_pred = net.predict(X)
    return r2_score(y, y_pred)


def rmse(net, X, y):
    y_pred = net.predict(X)
    return mean_squared_error(y, y_pred, squared=False)


def cal_metrics(y_true, y_pred, task_type, pred_prob=None, n_class=None, unique_labels=None):
    """
    y_prob, n_class, unique_labels are only used in classification task
    n_class: number of classes in real dataset
    unique_labels: unique labels in training dataset (thus its the dimension of the output of the model)
    """
    if task_type == "regression":
        r2 = r2_score(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        return {
            "r2": r2,
            "rmse": rmse,
        }
    else:
        # f1 only need to evaluate on seen classes
        y_true = y_true.astype(int)
        f1 = f1_score(y_true, y_pred, average="weighted")
        if task_type == "binary_classification":
            roc_auc = roc_auc_score(y_true, pred_prob[:, 1])
        else:
            # roc need to evaluate on all classes
            # fill the probability of unseen class to 0
            rest_label = set(range(n_class)) - set(unique_labels)
            tmp = []
            j = 0
            for i in range(n_class):
                if i in rest_label:
                    # unseen class, we set the probability to 0
                    tmp.append(np.array([0] * y_true.shape[0])[:, np.newaxis])
                else:
                    try:
                        tmp.append(pred_prob[:, [j]])
                    except:
                        tmp.append(pred_prob[:, np.newaxis])
                    j += 1
            filled_pred_prob = np.hstack(tmp)
            filled_y_true = np.eye(n_class)[y_true]
            # if no data in one class of y_true, roc_auc_score will raise error
            # see detail: https://github.com/scikit-learn/scikit-learn/issues/24636
            # get rid of the class with no data from both y_true and filled_pred_prob
            # remove the dimension of all 0
            index = (np.sum(filled_y_true, axis=0) > 0) & (np.sum(filled_pred_prob, axis=0) > 0)
            # np.save("filled_y_true.npy", filled_y_true)
            # np.save("filled_pred_prob.npy", filled_pred_prob)
            # np.save("index.npy", index)
            filled_y_true = filled_y_true[:, index]
            filled_pred_prob = filled_pred_prob[:, index]
            roc_auc = roc_auc_score(filled_y_true, filled_pred_prob, multi_class="ovr")

        return {
            "roc_auc": roc_auc,
            "f1": f1,
        }
