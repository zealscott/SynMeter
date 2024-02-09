import torch
import numpy as np
import os
from copy import deepcopy
from lib.commons import read_csv, improve_reproducibility
from .tablediff import train_wrapper_tablediffusion
import pandas as pd
import optuna
from lib.commons import preprocess, load_config, get_n_class
from lib.info import TUNED_PARAMS_PATH, NUMS_TRIALS, STORAGE
from lib.tune_helper import fidelity_tuner, utility_tuner


def train(args, cuda, seed=0):
    improve_reproducibility(seed)

    device = torch.device("cuda:" + cuda)

    model = train_wrapper_tablediffusion(args, device)

    # save training record and model
    path_params = args["path_params"]
    os.makedirs(os.path.dirname(path_params["loss_record"]), exist_ok=True)
    torch.save(model, path_params["out_model"])


def sample(args, n_samples=0, seed=0):
    improve_reproducibility(seed)

    path_params = args["path_params"]

    train_data_pd, meta_data, discrete_columns = read_csv(path_params["train_data"], path_params["meta_data"])
    val_data_pd, _, _ = read_csv(path_params["val_data"], path_params["meta_data"])
    # combine train and val data
    data_pd = pd.concat([train_data_pd, val_data_pd], ignore_index=True, sort=False)

    model = torch.load(path_params["out_model"])

    # sample the same number of data as the real data
    n_samples = n_samples if n_samples > 0 else len(data_pd)
    sampled = model.sample(n_samples)

    os.makedirs(os.path.dirname(path_params["out_data"]), exist_ok=True)
    # save synthetic data to csv
    sampled.to_csv(path_params["out_data"], index=False)


def _suggest_mlp_layers(trial):
    def suggest_dim(name):
        t = trial.suggest_int(name, d_min, d_max)
        return 2**t

    min_n_layers, max_n_layers, d_min, d_max = 1, 2, 5, 9
    n_layers = 2 * trial.suggest_int("n_layers", min_n_layers, max_n_layers)
    d_first = [suggest_dim("d_first")] if n_layers else []
    d_middle = [suggest_dim("d_middle")] * (n_layers - 2) if n_layers > 2 else []
    d_last = [suggest_dim("d_last")] if n_layers > 1 else []
    d_layers = d_first + d_middle + d_last
    return d_layers


def tune(config, cuda, dataset, seed=0):
    """
    tune tablediffusion using given config
    evaluate the ML performance of the synthetic data using fixed XGBoost
    train: synthetic data
    val: real data
    """

    def tablediffusion_objective(trial):
        # configure the model for this trail
        model_params = {}
        model_params["batch_size"] = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
        model_params["d_layers"] = _suggest_mlp_layers(trial)
        model_params["diffusion_steps"] = trial.suggest_categorical("diffusion_steps", [3, 5, 10, 15, 20])
        model_params["predict_noise"] = trial.suggest_categorical("predict_noise", [True, False])
        model_params["epoch_target"] = trial.suggest_categorical("epoch_target", [5, 10, 15, 20])
        model_params["lr"] = trial.suggest_float("lr", 0.0001, 0.01, log=True)

        model_params["epsilon_target"] = 1.0

        # store configures
        trial.set_user_attr("config", model_params)
        # train model with train+val data
        config["model_params"] = model_params
        try:
            model = train_wrapper_tablediffusion(config, device, tune=True)
            # sample and save the temporary synthetic data
            n_samples = meta_data["train_size"] + meta_data["val_size"]
            sampled = model.sample(n_samples)
            os.makedirs(os.path.dirname(path_params["out_data"]), exist_ok=True)
            sampled.to_csv(path_params["out_data"], index=False)

            # evaluate the temporary synthetic data
            fidelity = fidelity_tuner(config, seed)
            affinity, query_error = utility_tuner(config, dataset, cuda, seed)
            print("fidelity: {0}, affinity: {1}, query error: {2}".format(fidelity, affinity, query_error))
            error = fidelity + affinity + query_error
        except Exception as e:
            print("*" * 20 + "Error when tuning" + "*" * 20)
            print(e)
            error = 1e10
        return error

    device = torch.device("cuda:" + cuda)
    path_params = config["path_params"]

    # load real data
    real_train_data_pd, meta_data, discrete_columns = read_csv(path_params["train_data"], path_params["meta_data"])

    study_name = "tune_tablediffusion_{0}".format(dataset)
    try:
        optuna.delete_study(study_name=study_name, storage=STORAGE)
    except:
        pass
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=0),
        # storage=STORAGE,
        storage=None,
        study_name=study_name,
    )

    study.optimize(tablediffusion_objective, n_trials=NUMS_TRIALS, show_progress_bar=True)

    # update the best params
    config["model_params"] = study.best_trial.user_attrs["config"]
    config["sample_params"]["num_samples"] = meta_data["train_size"] + meta_data["val_size"] + meta_data["test_size"]
    config["sample_params"]["num_train_samples"] = meta_data["train_size"]
    config["sample_params"]["num_val_samples"] = meta_data["val_size"]

    print("best score for Tablediffusion {0}: {1}".format(dataset, study.best_value))

    return config
