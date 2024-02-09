import pandas as pd
from lib.commons import improve_reproducibility,read_csv
from .pgm import train_wrapper_PGM, reverse_data
from lib.info import STORAGE,N_EXPS
from lib.tune_helper import select_three_clique, select_two_clique
import os
import pickle
import optuna
from lib.tune_helper import fidelity_tuner, utility_tuner


def train(args, cuda, seed=0):
    improve_reproducibility(seed)

    model = train_wrapper_PGM(args, cuda)

    # save training record and model
    path_params = args["path_params"]
    os.makedirs(os.path.dirname(path_params["out_model"]), exist_ok=True)
    pickle.dump(model, open(path_params["out_model"], "wb"))


def sample(args, n_samples=0, seed=0):
    improve_reproducibility(seed)

    path_params = args["path_params"]

    train_data_pd, meta_data, discrete_columns = read_csv(path_params["train_data"], path_params["meta_data"])
    val_data_pd, _, _ = read_csv(path_params["val_data"], path_params["meta_data"])
    # combine train and val data
    data_pd = pd.concat([train_data_pd, val_data_pd], ignore_index=True, sort=False)

    model = pickle.load(open(path_params["out_model"], "rb"))
    learned_pgm = model["learned_pgm"]
    supports = model["supports"]
    data_transformer = model["data_transformer"]

    # sample the same number of data as the real data
    n_samples = n_samples if n_samples > 0 else len(data_pd)
    synth = learned_pgm.synthetic_data(rows=n_samples)

    syn_data = reverse_data(synth, supports)

    sampled = data_transformer.inverse_transform(syn_data.df)

    os.makedirs(os.path.dirname(path_params["out_data"]), exist_ok=True)
    # save synthetic data to csv
    sampled.to_csv(path_params["out_data"], index=False)


def tune(config, cuda, dataset, seed=0):
    """
    tune MST
    """
    marginal_nums = [50, 10]

    def mst_objective(trial):
        # configure the model for this trail
        model_params = {}
        model_params["num_iters"] = 5000
        model_params["max_bins"] = 10  # more bins will slow down the training
        model_params["epsilon"] = 30000000.0  # infinite privacy budget
        model_params["delta"] = 1e-9
        model_params["2_cliques"] = select_two_clique(real_train_data_pd.columns, n=marginal_nums[0])
        model_params["3_cliques"] = select_three_clique(real_train_data_pd.columns, n=marginal_nums[1])
        model_params["bi_nums"] = len(model_params["2_cliques"])
        model_params["tri_nums"] = len(model_params["3_cliques"])

        # store configures
        trial.set_user_attr("config", model_params)
        config["model_params"] = model_params

        try:
            # train the model
            model = train_wrapper_PGM(config, cuda, tune=True)
            learned_pgm = model["learned_pgm"]
            supports = model["supports"]
            data_transformer = model["data_transformer"]

            # sample synthetic data
            n_samples = meta_data["train_size"] + meta_data["val_size"]
            synth = learned_pgm.synthetic_data(rows=n_samples)
            syn_data = reverse_data(synth, supports)
            sampled = data_transformer.inverse_transform(syn_data.df)
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
            marginal_nums[1] = max(0, marginal_nums[1] - 5)
            if marginal_nums[1] == 0:
                marginal_nums[0] = max(0, marginal_nums[0] - 5)

        return error

    path_params = config["path_params"]
    # load real data
    real_train_data_pd, meta_data, discrete_columns = read_csv(path_params["train_data"], path_params["meta_data"])

    study_name = "tune_mst_{0}".format(dataset)
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

    study.optimize(mst_objective, n_trials=5, show_progress_bar=True)

    # update the best params
    config["model_params"] = study.best_trial.user_attrs["config"]
    config["sample_params"]["num_samples"] = meta_data["train_size"] + meta_data["val_size"] + meta_data["test_size"]
    config["sample_params"]["num_train_samples"] = meta_data["train_size"]
    config["sample_params"]["num_val_samples"] = meta_data["val_size"]

    print("best score for MST {0}: {1}".format(dataset, study.best_value))

    return config
