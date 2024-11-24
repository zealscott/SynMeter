from realtabformer import REaLTabFormer
from lib.commons import read_csv, improve_reproducibility, preprocess
import torch
import os
import optuna
from lib.info import TUNED_PARAMS_PATH, NUMS_TRIALS, STORAGE
from lib.commons import load_config, get_n_class
import pandas as pd
import optuna


def init_model(model_params):
    model = REaLTabFormer(
        model_type="tabular",
        gradient_accumulation_steps=model_params["gradient_accumulation_steps"],
        logging_steps=100,
        epochs=100,
    )

    return model


def train(args, cuda, seed=0):
    """
    train GreaT using given args
    used when the best parameters are found and stored in `exp/`
    transformers typically use all available GPUs
    """
    improve_reproducibility(seed)

    model_params = args["model_params"]
    path_params = args["path_params"]

    train_data_pd, meta_data, discrete_columns = read_csv(path_params["train_data"], path_params["meta_data"])
    val_data_pd, _, _ = read_csv(path_params["val_data"], path_params["meta_data"])
    # combine train and val data
    data_pd = pd.concat([train_data_pd, val_data_pd], ignore_index=True, sort=False)

    model = init_model(model_params)
    os.makedirs(os.path.dirname(path_params["out_model"]), exist_ok=True)
    print(f"Saving model to {path_params['out_model']}")
    saved_dir = os.path.dirname(path_params["out_model"])
    # model.experiment_id = "0"
    model.fit(data_pd, n_critic=0)
    model.save(saved_dir)


def sample(args, n_samples=0, seed=0):
    """
    sample synthetic data from the loaded CTGAN
    used when the model is already trained and saved
    """
    improve_reproducibility(seed)

    model_params = args["model_params"]
    path_params = args["path_params"]

    train_data_pd, meta_data, discrete_columns = read_csv(path_params["train_data"], path_params["meta_data"])
    val_data_pd, _, _ = read_csv(path_params["val_data"], path_params["meta_data"])
    # combine train and val data
    data_pd = pd.concat([train_data_pd, val_data_pd], ignore_index=True, sort=False)
    # sample the same number of data as the real data
    n_samples = n_samples if n_samples > 0 else len(data_pd)

    # load model
    # get model save path (exclude the last part of the path)
    saved_dir = os.path.dirname(path_params["out_model"])

    model = init_model(model_params)
    # model.experiment_id = "0"
    model = model.load_from_dir(saved_dir)

    sampled = model.sample(n_samples)

    # remove space in column names and values with strip
    # Function to convert 'x.' to float
    def convert_to_float(value):
        if isinstance(value, str) and value.endswith('.'):
            return float(value[:-1])
        if isinstance(value, str) and value == '':
            return float(0)
        return value

    # remove space in column names and values with strip
    sampled.columns = sampled.columns.str.strip()
    sampled = sampled.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # Apply the function to the entire DataFrame
    sampled = sampled.applymap(convert_to_float)

    os.makedirs(os.path.dirname(path_params["out_data"]), exist_ok=True)
    # save synthetic data to csv
    sampled.to_csv(path_params["out_data"], index=False)


def tune(config, cuda, dataset, seed=0):
    """
    use the default hyperparameters for each dataset
    """

    path_params = config["path_params"]

    # load real data
    real_train_data_pd, meta_data, discrete_columns = read_csv(path_params["train_data"], path_params["meta_data"])
    real_val_data_pd, _, _ = read_csv(path_params["val_data"], path_params["meta_data"])

    # update the best params
    config["sample_params"]["num_samples"] = meta_data["train_size"] + meta_data["val_size"] + meta_data["test_size"]
    config["sample_params"]["num_train_samples"] = meta_data["train_size"]
    config["sample_params"]["num_val_samples"] = meta_data["val_size"]

    return config
