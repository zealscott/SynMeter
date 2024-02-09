from lib.commons import load_json
import pandas as pd
import numpy as np
from evaluator.fidelity.wasserstein import cal_fidelity
import os
import json


def load_data_for_fidelity(data_path, meta_data_path):
    """
    load raw data, meta data and discrete columns for fidelity evaluation
    """
    data = pd.read_csv(
        data_path, dtype=object
    )  # use objective for easily compare, otherwise int and str will be different
    meta_data = load_json(meta_data_path)

    # construct column to value dict
    col_value_dict = {}
    for col in meta_data["columns"]:
        if col["type"] != "continuous":
            col_value_dict[col["name"]] = col["i2s"]
            # make sure the data type is string
            data[col["name"]] = data[col["name"]].astype(str)
        else:
            # make sure the data type is numeric
            data[col["name"]] = pd.to_numeric(data[col["name"]])
    return data, col_value_dict


def fidelity_evaluation(args, seed, tune=False, eval_type="test"):
    """
    evaluate the fidelity for synthetic data with wassterstein distance
    """
    path_params = args["path_params"]
    # load all real data
    if eval_type == "train":
        real_data_path = path_params["train_data"]
    else:
        real_data_path = path_params["val_data"] if tune else path_params["test_data"]

    real_data, dis_col_value_dict = load_data_for_fidelity(real_data_path, path_params["meta_data"])
    syn_data, _ = load_data_for_fidelity(path_params["out_data"], path_params["meta_data"])

    # sample syn_data to make it have the same number of rows as real_data
    syn_data = syn_data.sample(len(real_data), replace=False, random_state=seed)

    if tune and len(real_data) > 2000:
        num_cols = len(real_data.columns)
        k = 1000 if num_cols > 40 else 2000
        real_data = real_data.sample(k)
        syn_data = syn_data.sample(k)
    elif len(real_data) > 5000:
        # sample 5k for evaluation
        real_data = real_data.sample(3000)
        syn_data = syn_data.sample(3000)
    res = cal_fidelity(real_data, syn_data, dis_col_value_dict)

    return res


def add_fidelity_results(cur_res, fidelity_res):
    if not fidelity_res:
        for metric, value in cur_res.items():
            fidelity_res[metric] = [value]
    else:
        for metric, value in cur_res.items():
            fidelity_res[metric].append(value)
    return fidelity_res


def save_fidelity_results(fidelity_res, fidelity_result_path):
    """
    save the stat result with json
    """
    # get the average, std result use numpy
    for metric, value in fidelity_res.items():
        fidelity_res[metric] = {}
        fidelity_res[metric]["mean"] = sum(value) / len(value)
        fidelity_res[metric]["std"] = float(np.std(value))

    # save the result with json
    os.makedirs(os.path.dirname(fidelity_result_path), exist_ok=True)
    with open(fidelity_result_path, "w") as f:
        json.dump(fidelity_res, f)

    print("fidelity result saved to {}".format(fidelity_result_path))
