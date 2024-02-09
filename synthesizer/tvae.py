# adapt from https://github.com/sdv-dev/TVAE/tree/master
from .gan import TVAE
from lib.commons import read_csv, improve_reproducibility, get_n_class
import torch
import os
import pandas as pd
from lib.info import NUMS_TRIALS, STORAGE
import optuna
from lib.tune_helper import fidelity_tuner, utility_tuner

def init_model(model_params, device):
    model = TVAE(
        embedding_dim=model_params["embedding_dim"],
        compress_dims=model_params["compress_dims"],
        decompress_dims=model_params["decompress_dims"],
        l2scale=model_params["l2scale"],
        batch_size=model_params["batch_size"],
        epochs=model_params["epochs"],
        loss_factor=model_params["loss_factor"],
        cuda=device,
    )

    return model


def train(args, cuda, seed=0):
    """
    train TVAE using given args
    used when the best parameters are found and stored in `exp/`
    """
    improve_reproducibility(seed)

    device = torch.device("cuda:" + cuda)
    model_params = args["model_params"]
    path_params = args["path_params"]

    train_data_pd, meta_data, discrete_columns = read_csv(path_params["train_data"], path_params["meta_data"])
    val_data_pd, _, _ = read_csv(path_params["val_data"], path_params["meta_data"])
    # combine train and val data
    data_pd = pd.concat([train_data_pd, val_data_pd], ignore_index=True, sort=False)

    model = init_model(model_params, device)

    model.fit(data_pd, discrete_columns)

    # save training record and model
    os.makedirs(os.path.dirname(path_params["loss_record"]), exist_ok=True)
    model.loss_history.to_csv(path_params["loss_record"], index=False)
    torch.save(model, path_params["out_model"])


def sample(args, n_samples=0, seed=0):
    """
    sample synthetic data from the loaded TVAE
    used when the model is already trained and saved
    """
    improve_reproducibility(seed)

    model_params = args["model_params"]
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


def tune(config, cuda, dataset, seed=0):
    def tvae_objective(trial):
        # configure the model for this trail
        model_params = {}
        model_params["epochs"] = trial.suggest_int("epochs", 100, 500)
        model_params["batch_size"] = trial.suggest_categorical("batch_size", [500, 1000, 2000, 5000])
        model_params["l2scale"] = trial.suggest_float("l2scale", 1e-6, 1e-3, log=True)
        model_params["loss_factor"] = trial.suggest_float("loss_factor", 1.0, 5.0)

        model_params["embedding_dim"] = 128
        model_params["compress_dims"] = [128, 128]
        model_params["decompress_dims"] = [128, 128]

        # store configures
        trial.set_user_attr("config", model_params)
        config["model_params"] = model_params

        # train model
        model = init_model(model_params, device)
        model.fit(real_train_data_pd, discrete_columns)
        
        try:
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

    study_name = "tune_tvae_{0}".format(dataset)
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

    study.optimize(tvae_objective, n_trials=NUMS_TRIALS, show_progress_bar=True)

    # update the best params
    config["model_params"] = study.best_trial.user_attrs["config"]
    config["sample_params"]["num_samples"] = meta_data["train_size"] + meta_data["val_size"] + meta_data["test_size"]
    config["sample_params"]["num_train_samples"] = meta_data["train_size"]
    config["sample_params"]["num_val_samples"] = meta_data["val_size"]

    print("best score for TVAE {0}: {1}".format(dataset, study.best_value))

    return config
