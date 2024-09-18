from lib.commons import read_csv, improve_reproducibility
import torch
import os
import pandas as pd
from synthesizer.tab_syn.vae_main import train_vae
from synthesizer.tab_syn.diffusion_main import train_diffusion
from synthesizer.tab_syn.sample import sample_from_tabsyn


def train(args, cuda, seed=0):
    """
    train TabSyn using given args
    used when the best parameters are found and stored in `exp/`
    """
    improve_reproducibility(seed)

    device = torch.device("cuda:" + cuda)

    # test whethet train_z exists, if not, train VAE first
    saved_dir = os.path.dirname(args["path_params"]["out_model"])
    if not os.path.exists(f"{saved_dir}/train_z.npy"):
        train_vae(args, device)

    train_diffusion(args, device)


def sample(args, n_samples=0, seed=0):
    """
    sample synthetic data from the loaded CTGAN
    used when the model is already trained and saved
    """
    improve_reproducibility(seed)
    path_params = args["path_params"]

    train_data_pd, meta_data, discrete_columns = read_csv(
        path_params["train_data"], path_params["meta_data"]
    )
    val_data_pd, _, _ = read_csv(path_params["val_data"], path_params["meta_data"])
    # combine train and val data
    data_pd = pd.concat([train_data_pd, val_data_pd], ignore_index=True, sort=False)

    # sample the same number of data as the real data
    n_samples = n_samples if n_samples > 0 else len(data_pd)

    sampled = sample_from_tabsyn(args, n_samples, "cuda:0")

    os.makedirs(os.path.dirname(path_params["out_data"]), exist_ok=True)
    # save synthetic data to csv
    sampled.to_csv(path_params["out_data"], index=False)


def tune(config, cuda, dataset, seed=0):
    """
    TabSyn is robust to hyperparameters
    use the default hyperparameters for each dataset
    """
    path_params = config["path_params"]

    # load real data
    real_train_data_pd, meta_data, discrete_columns = read_csv(
        path_params["train_data"], path_params["meta_data"]
    )
    real_val_data_pd, _, _ = read_csv(path_params["val_data"], path_params["meta_data"])

    # update the best params
    config["sample_params"]["num_samples"] = (
        meta_data["train_size"] + meta_data["val_size"] + meta_data["test_size"]
    )
    config["sample_params"]["num_train_samples"] = meta_data["train_size"]
    config["sample_params"]["num_val_samples"] = meta_data["val_size"]

    return config
