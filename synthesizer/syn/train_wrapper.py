from pathlib import Path
from loguru import logger
import numpy as np
from lib.commons import read_csv
import pandas as pd
from .data_trasnformer import DataTransformer
from .data_loader import DataLoader
from .dpsyn import DPSyn


def train_wrapper_privsyn(args, tune = False):
    path_params = args["path_params"]
    model_params = args["model_params"]

    epsilon = model_params["epsilon"]
    delta = model_params["delta"]
    max_bins = model_params["max_bins"]
    update_iterations = model_params["update_iterations"]
    
    # budget allocation for DP
    ratio = model_params["ratio"] if "ratio" in model_params else None

    # prepare data
    train_data_pd, meta_data, discrete_columns = read_csv(
        path_params["train_data"], path_params["meta_data"]
    )
    val_data_pd, _, _ = read_csv(path_params["val_data"], path_params["meta_data"])
    if tune:
        data_pd = train_data_pd
    else:
        # combine train and val data
        data_pd = pd.concat([train_data_pd, val_data_pd], ignore_index=True, sort=False)

    data_transformer = DataTransformer(max_bins)

    transformed_data = data_transformer.fit_transform(data_pd, discrete_columns)
    encode_mapping = data_transformer.get_mapping()

    # dataloader initialization
    data_loader = DataLoader()
    data_loader.load_data(private_data=transformed_data, encode_mapping=encode_mapping)

    synthesizer = DPSyn(data_loader, update_iterations, epsilon, delta, sensitivity=1, ratio=ratio)
    synthesizer.train()
    
    model = {}
    model["learned_privsyn"] = synthesizer
    model["data_transformer"] = data_transformer
    model["data_loader"] = data_loader

    return model
    