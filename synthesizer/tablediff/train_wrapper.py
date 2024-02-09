import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from copy import deepcopy
from lib.commons import read_csv
from .modules import TableDiffusion_Synthesiser


def train_wrapper_tablediffusion(args, device, tune=False):
    model_params = args["model_params"]
    path_params = args["path_params"]

    batch_size = model_params["batch_size"]
    lr = model_params["lr"]
    diffusion_steps = model_params["diffusion_steps"]
    predict_noise = model_params["predict_noise"]
    epsilon_target = model_params["epsilon_target"]
    epoch_target = model_params["epoch_target"]  # epoch for determining epsilon
    dim = model_params["d_layers"]
    device = device

    # prepare data
    train_data_pd, meta_data, discrete_columns = read_csv(path_params["train_data"], path_params["meta_data"])
    val_data_pd, _, _ = read_csv(path_params["val_data"], path_params["meta_data"])
    if tune:
        data_pd = train_data_pd
    else:
        # combine train and val data
        data_pd = pd.concat([train_data_pd, val_data_pd], ignore_index=True, sort=False)

    model = TableDiffusion_Synthesiser(
        batch_size=batch_size,
        lr=lr,
        dims=dim,
        diffusion_steps=diffusion_steps,
        predict_noise=predict_noise,
        epsilon_target=epsilon_target,
        epoch_target=epoch_target,
        device=device,
    )

    model.fit(df=data_pd, discrete_columns=discrete_columns)

    return model
