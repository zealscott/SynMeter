import pandas as pd
from lib.commons import load_config, improve_reproducibility, read_csv
from .syn import train_wrapper_privsyn, RecordPostprocessor
import os
import pickle
import optuna
from lib.tune_helper import fidelity_tuner, utility_tuner
from lib.info import *

def train(args, cuda, seed=0):
    improve_reproducibility(seed)

    model = train_wrapper_privsyn(args)

    # save training record and model
    path_params = args["path_params"]
    os.makedirs(os.path.dirname(path_params["out_model"]), exist_ok=True)
    pickle.dump(model, open(path_params["out_model"], "wb"))


def sample(args, n_samples=0, seed=0):
    improve_reproducibility(seed)

    path_params = args["path_params"]

    train_data_pd, meta_data, discrete_columns = read_csv(
        path_params["train_data"], path_params["meta_data"]
    )
    val_data_pd, _, _ = read_csv(path_params["val_data"], path_params["meta_data"])
    # combine train and val data
    data_pd = pd.concat([train_data_pd, val_data_pd], ignore_index=True, sort=False)

    model = pickle.load(open(path_params["out_model"], "rb"))
    learned_privsyn = model["learned_privsyn"]
    data_transformer = model["data_transformer"]
    data_loader = model["data_loader"]

    # sample the same number of data as the real data
    n_samples = n_samples if n_samples > 0 else len(data_pd)
    syn_data = learned_privsyn.synthesize(num_records=n_samples)
    
    syn_data = data_transformer.inverse_transform(syn_data)

    # # post-processing generated data, map records with grouped/binned attribute back to original attributes
    # print("********************* START POSTPROCESSING ***********************")
    # postprocessor = RecordPostprocessor()
    # syn_data = postprocessor.post_process(syn_data, data_loader.decode_mapping)
    # syn_data = syn_data[data_pd.columns]
    # print("------------------------>synthetic data post-processed:")
    # print(syn_data)

    os.makedirs(os.path.dirname(path_params["out_data"]), exist_ok=True)
    # save synthetic data to csv
    syn_data.to_csv(path_params["out_data"], index=False)


def tune(config, cuda, dataset, seed=0):
    """
    tune privsyn
    """
    def privsyn_objective(trial):
        # configure the model for this trial
        model_params = {}
        model_params["epsilon"] = 100000000.0
        model_params["delta"] = 3.4498908254380166e-11
        model_params["max_bins"] = trial.suggest_int("max_bins", 10, 50)
        model_params["update_iterations"] = trial.suggest_int("update_iterations", 10, 100)
        
        # store configures
        trial.set_user_attr("config", model_params)
        config["model_params"] = model_params
        
        try:
            # train the model
            model = train_wrapper_privsyn(config, tune=True)
            learned_privsyn = model["learned_privsyn"]
            data_transformer = model["data_transformer"]
            
            # sample synthetic data
            n_samples = meta_data["train_size"] + meta_data["val_size"]
            syn_data = learned_privsyn.synthesize(num_records=n_samples)
            sampled = data_transformer.inverse_transform(syn_data)
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

    study_name = "tune_privsyn_{0}".format(dataset)
    try:
        optuna.delete_study(study_name=study_name, storage=STORAGE)
    except:
        pass
    study = optuna.create_study(
        direction="minimize" ,
        sampler=optuna.samplers.TPESampler(seed=0),
        # storage=STORAGE,
        study_name=study_name,
    )

    study.optimize(privsyn_objective, n_trials=10, show_progress_bar=True)

    # update the best params
    config["model_params"] = study.best_trial.user_attrs["config"]
    config["sample_params"]["num_samples"] = meta_data["train_size"] + meta_data["val_size"] + meta_data["test_size"]
    config["sample_params"]["num_train_samples"] = meta_data["train_size"]
    config["sample_params"]["num_val_samples"] = meta_data["val_size"]
    
    print("best score for privsyn {0}: {1}".format(dataset, study.best_value))

    return config
