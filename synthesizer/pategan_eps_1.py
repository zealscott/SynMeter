from synthcity.plugins import Plugins
from synthcity.utils.serialization import save_to_file, load_from_file
from synthcity.plugins.core.dataloader import GenericDataLoader
from lib.commons import read_csv, improve_reproducibility
import torch
import os
import optuna
from lib.info import NUMS_TRIALS, STORAGE
import pandas as pd
import optuna
from lib.tune_helper import fidelity_tuner, utility_tuner


def train(args, cuda, seed=0):
    """
    train CTGAN using given args
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

    loader = GenericDataLoader(data_pd)

    model = Plugins().get("pategan", **model_params, device=device)

    model.fit(loader)

    # save training record and model
    os.makedirs(os.path.dirname(path_params["loss_record"]), exist_ok=True)
    save_to_file(path_params["out_model"], model)


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
    model = load_from_file(path_params["out_model"])

    # sample the same number of data as the real data
    n_samples = n_samples if n_samples > 0 else len(data_pd)
    sampled = model.generate(n_samples).dataframe()

    os.makedirs(os.path.dirname(path_params["out_data"]), exist_ok=True)
    # save synthetic data to csv
    sampled.to_csv(path_params["out_data"], index=False)


def tune(config, cuda, dataset, seed=0):
    def ctgan_objective(trial):
        # configure the model for this trail
        model_params = {}
        model_params["n_iter"] = trial.suggest_int("n_iter", 1000, 5000)
        model_params["generator_n_layers_hidden"] = trial.suggest_int("generator_n_layers_hidden", 1, 3)
        model_params["generator_n_units_hidden"] = trial.suggest_int("generator_n_units_hidden", 50, 200)
        model_params["discriminator_n_layers_hidden"] = trial.suggest_int("discriminator_n_layers_hidden", 1, 3)
        model_params["discriminator_n_units_hidden"] = trial.suggest_int("discriminator_n_units_hidden", 50, 200)
        model_params["n_teachers"] = trial.suggest_int("n_teachers", 5, 20)
        model_params["lr"] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

        model_params["epsilon"] = 1.0

        # store configures
        trial.set_user_attr("config", model_params)
        config["model_params"] = model_params

        # train model
        loader = GenericDataLoader(real_train_data_pd)
        model = Plugins().get("pategan", **model_params, device=device)
        model.fit(loader)

        try:
            # sample and save the temporary synthetic data
            n_samples = meta_data["train_size"] + meta_data["val_size"]
            sampled = model.generate(n_samples).dataframe()
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

    study_name = "tune_pategan_{0}".format(dataset)
    try:
        optuna.delete_study(study_name=study_name, storage=STORAGE)
    except:
        pass
    study = optuna.create_study(
        direction="minimize",  # minimize the error from the fidelity and utility
        sampler=optuna.samplers.TPESampler(seed=0),
        # storage=STORAGE,
        storage=None,
        study_name=study_name,
    )

    study.optimize(ctgan_objective, n_trials=NUMS_TRIALS, show_progress_bar=True)

    # update the best params
    config["model_params"] = study.best_trial.user_attrs["config"]
    config["sample_params"]["num_samples"] = meta_data["train_size"] + meta_data["val_size"] + meta_data["test_size"]
    config["sample_params"]["num_train_samples"] = meta_data["train_size"]
    config["sample_params"]["num_val_samples"] = meta_data["val_size"]

    print("best score for PATEGAN {0}: {1}".format(dataset, study.best_value))

    return config
