import torch
import numpy as np
import os
from copy import deepcopy
from lib.commons import read_csv, improve_reproducibility
from .ddpm import train_wrapper_tab_ddpm
import pandas as pd
import optuna
from lib.commons import preprocess, load_config, get_n_class
from lib.info import TUNED_PARAMS_PATH, NUMS_TRIALS, STORAGE
from lib.tune_helper import fidelity_tuner, utility_tuner


def train(args, cuda, seed=0):
    improve_reproducibility(seed)

    device = torch.device("cuda:" + cuda)

    trainer, _ = train_wrapper_tab_ddpm(args, device)

    trainer.run_loop()

    # save training record and model
    path_params = args["path_params"]
    os.makedirs(os.path.dirname(path_params["loss_record"]), exist_ok=True)
    trainer.loss_history.to_csv(path_params["loss_record"], index=False)
    torch.save(trainer.diffusion, path_params["out_model"])


def sample(args, n_samples=0, seed=0):
    improve_reproducibility(seed)

    model_params = args["model_params"]
    path_params = args["path_params"]

    train_data_pd, meta_data, discrete_columns = read_csv(path_params["train_data"], path_params["meta_data"])
    val_data_pd, _, _ = read_csv(path_params["val_data"], path_params["meta_data"])
    # combine train and val data
    data_pd = pd.concat([train_data_pd, val_data_pd], ignore_index=True, sort=False)

    model = torch.load(path_params["out_model"])
    data_transformer = model.data_transformer

    # sample the same number of data as the real data
    n_samples = n_samples if n_samples > 0 else len(data_pd)
    empirical_class_dist = data_transformer.empirical_class_dist

    gen_x, gen_y = model.sample_all(n_samples, batch_size=20000, y_dist=empirical_class_dist)

    sampled = data_transformer.inverse_transform(gen_x, gen_y)

    os.makedirs(os.path.dirname(path_params["out_data"]), exist_ok=True)
    # save synthetic data to csv
    sampled.to_csv(path_params["out_data"], index=False)


def _suggest_mlp_layers(trial):
    def suggest_dim(name):
        t = trial.suggest_int(name, d_min, d_max)
        return 2**t

    min_n_layers, max_n_layers, d_min, d_max = 1, 4, 7, 10
    n_layers = 2 * trial.suggest_int("n_layers", min_n_layers, max_n_layers)
    d_first = [suggest_dim("d_first")] if n_layers else []
    d_middle = [suggest_dim("d_middle")] * (n_layers - 2) if n_layers > 2 else []
    d_last = [suggest_dim("d_last")] if n_layers > 1 else []
    d_layers = d_first + d_middle + d_last
    return d_layers


def tune(config, cuda, dataset, seed=0):
    """
    tune TabDDPM using given config
    evaluate the ML performance of the synthetic data using fixed XGBoost
    train: synthetic data
    val: real data
    """
    def tabddpm_objective(trial):
        # configure the model for this trail
        model_params = {}
        model_params["lr"] = trial.suggest_float("lr", 0.00001, 0.003, log=True)
        model_params["steps"] = trial.suggest_categorical("steps", [5000, 20000, 30000])
        model_params["num_timesteps"] = trial.suggest_categorical("num_timesteps", [100, 1000])
        model_params["batch_size"] = trial.suggest_categorical("batch_size", [256, 4096])
        model_params["d_layers"] = _suggest_mlp_layers(trial)

        model_params["dropout"] = 0.0
        model_params["weight_decay"] = 0.0

        # store configures
        trial.set_user_attr("config", model_params)
        # train model with train+val data
        config["model_params"] = model_params
        trainer, _ = train_wrapper_tab_ddpm(config, device, tune=True)
        trainer.run_loop()
        # load trained model
        model = trainer.diffusion
        data_transformer = model.data_transformer

        try:
            # sample and save the temporary synthetic data
            n_samples = meta_data["train_size"] + meta_data["val_size"]
            empirical_class_dist = data_transformer.empirical_class_dist
            gen_x, gen_y = model.sample_all(n_samples, batch_size=10000, y_dist=empirical_class_dist)
            sampled = data_transformer.inverse_transform(gen_x, gen_y)
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

    study_name = "tune_tabddpm_{0}".format(dataset)
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

    study.optimize(tabddpm_objective, n_trials=NUMS_TRIALS, show_progress_bar=True)

    # update the best params
    config["model_params"] = study.best_trial.user_attrs["config"]
    config["sample_params"]["num_samples"] = meta_data["train_size"] + meta_data["val_size"] + meta_data["test_size"]
    config["sample_params"]["num_train_samples"] = meta_data["train_size"]
    config["sample_params"]["num_val_samples"] = meta_data["val_size"]

    print("best score for TabDDPM {0}: {1}".format(dataset, study.best_value))

    return config
