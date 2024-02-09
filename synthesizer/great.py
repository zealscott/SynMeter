# adapt from https://github.com/sdv-dev/CTGAN/tree/master
from .be_great import GReaT
from lib.commons import read_csv, improve_reproducibility
import os
import optuna
from lib.info import STORAGE
import pandas as pd
import optuna
from lib.tune_helper import fidelity_tuner, utility_tuner


def init_model(model_params, saved_dir):
    model = GReaT(
        llm="gpt2",
        epochs=model_params["epochs"],
        batch_size=model_params["batch_size"],
        experiment_dir=saved_dir,
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

    train_data_pd, meta_data, discrete_columns = read_csv(
        path_params["train_data"], path_params["meta_data"]
    )
    val_data_pd, _, _ = read_csv(path_params["val_data"], path_params["meta_data"])
    # combine train and val data
    data_pd = pd.concat([train_data_pd, val_data_pd], ignore_index=True, sort=False)
    # exclude discrete columns to get the continuous columns
    continuous_columns = [col for col in data_pd.columns if col not in discrete_columns]
    # get model save path (exclude the last part of the path)
    saved_dir = os.path.dirname(path_params["out_model"])

    model = init_model(model_params, saved_dir)

    trainer = model.fit(data_pd)

    # save training record and model
    loss_hist = trainer.state.log_history.copy()
    os.makedirs(os.path.dirname(path_params["loss_record"]), exist_ok=True)
    # loss hist is list of dict, convert to dataframe
    loss_hist = pd.DataFrame(loss_hist)
    loss_hist.to_csv(path_params["loss_record"], index=False)

    model.save(saved_dir, path_params["out_model"])


def sample(args, n_samples=0, seed=0):
    """
    sample synthetic data from the loaded CTGAN
    used when the model is already trained and saved
    """
    improve_reproducibility(seed)

    model_params = args["model_params"]
    path_params = args["path_params"]

    train_data_pd, meta_data, discrete_columns = read_csv(
        path_params["train_data"], path_params["meta_data"]
    )
    val_data_pd, _, _ = read_csv(path_params["val_data"], path_params["meta_data"])
    # combine train and val data
    data_pd = pd.concat([train_data_pd, val_data_pd], ignore_index=True, sort=False)
    # sample the same number of data as the real data
    n_samples = n_samples if n_samples > 0 else len(data_pd)

    # load model
    # get model save path (exclude the last part of the path)
    saved_dir = os.path.dirname(path_params["out_model"])
    model = GReaT.load_from_dir(saved_dir, path_params["out_model"])

    sampled = model.sample(
        n_samples, k=100, device="cuda:1", temperature=model_params["temperature"]
    )

    # remove space in column names and values with strip
    sampled.columns = sampled.columns.str.strip()
    sampled = sampled.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    os.makedirs(os.path.dirname(path_params["out_data"]), exist_ok=True)
    # save synthetic data to csv
    sampled.to_csv(path_params["out_data"], index=False)


def tune(config, cuda, dataset, seed=0):
    """
    use the default hyperparameters for each dataset
    """

    def great_objective(trial):
        # configure the model for this trail
        model_params = {}
        model_params["epochs"] = trial.suggest_int("epochs", 100,300)
        model_params["batch_size"] = trial.suggest_categorical(
            "batch_size", [8, 16, 32]
        )
        model_params["temperature"] = trial.suggest_float("temperature", 0.6, 0.9)

        # store configures
        trial.set_user_attr("config", model_params)
        config["model_params"] = model_params

        # train model
        model = init_model(model_params, saved_dir)
        try:
            trainer = model.fit(real_train_data_pd)
            # sample and save the temporary synthetic data
            n_samples = meta_data["train_size"] + meta_data["val_size"]
            sampled = model.sample(n_samples, k=100, device="cuda:1", temperature=model_params["temperature"])
            if len(sampled) != n_samples:
                # error 
                raise Exception("cannot sample enough data")
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

    path_params = config["path_params"]

    # load real data
    real_train_data_pd, meta_data, discrete_columns = read_csv(
        path_params["train_data"], path_params["meta_data"]
    )

    # configuration
    # get model save path (exclude the last part of the path)
    saved_dir = os.path.dirname(path_params["out_model"])

    study_name = "tune_great_{0}".format(dataset)
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

    # we can only tune 5 times due to computational cost
    study.optimize(great_objective, n_trials=5, show_progress_bar=True)

    # update the best params
    config["model_params"] = study.best_trial.user_attrs["config"]
    config["sample_params"]["num_samples"] = (
        meta_data["train_size"] + meta_data["val_size"] + meta_data["test_size"]
    )
    config["sample_params"]["num_train_samples"] = meta_data["train_size"]
    config["sample_params"]["num_val_samples"] = meta_data["val_size"]

    print("best score for GreaT {0}: {1}".format(dataset, study.best_value))

    return config
