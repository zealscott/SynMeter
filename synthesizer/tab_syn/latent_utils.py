import os
import json
import numpy as np
import pandas as pd
import torch
from .vae_model import Decoder_model
import pickle

def get_input_train(args):
    path_params = args["path_params"]
    # get the saved directory
    out_dir = os.path.dirname(path_params["out_model"])
    embedding_save_path = f"{out_dir}/train_z.npy"
    train_z = torch.tensor(np.load(embedding_save_path)).float()

    train_z = train_z[:, 1:, :] # remove the first token
    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim

    # flatten the input
    train_z = train_z.view(B, in_dim)

    return train_z, out_dir


def get_input_generate(args):
    saved_dir = os.path.dirname(args["path_params"]["out_model"])

    embedding_save_path = f"{saved_dir}/train_z.npy"
    train_z = torch.tensor(np.load(embedding_save_path)).float()

    train_z = train_z[:, 1:, :]

    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim

    train_z = train_z.view(B, in_dim)
    
    with open(f"{saved_dir}/data_transformer.pkl", "rb") as f:
        data_transformer = pickle.load(f)
    
    d_numerical = data_transformer.get_num_dim()
    categories = data_transformer.get_cat_dim()
    
    pre_decoder = Decoder_model(2, d_numerical, categories, 4, n_head=1, factor=32)

    decoder_save_path = f"{saved_dir}/decoder.pt"
    pre_decoder.load_state_dict(torch.load(decoder_save_path))

    return train_z, args["path_params"]["out_model"], data_transformer, pre_decoder, token_dim


@torch.no_grad()
def recover_data(syn_data, data_transformer,pre_decoder, token_dim, device):
    """
    recover data from vae, then inverse transform the data
    """
    syn_data = syn_data.reshape(syn_data.shape[0], -1, token_dim)
    norm_input = pre_decoder(torch.tensor(syn_data).to(device))
    x_hat_num, x_hat_cat = norm_input

    syn_cat = []
    for pred in x_hat_cat:
        # get the one with the highest probability
        syn_cat.append(pred.argmax(dim=-1))


    syn_num = x_hat_num.cpu().numpy() if x_hat_num is not None else None
    syn_cat = torch.stack(syn_cat).t().cpu().numpy() if syn_cat else None

    if syn_num is None:
        syn_numpy = syn_cat
    elif syn_cat is None:
        syn_numpy = syn_num
    else:
        # concatenate the numerical and categorical data
        syn_num = syn_num.reshape(syn_num.shape[0], -1)
        syn_cat = syn_cat.reshape(syn_cat.shape[0], -1)
        syn_numpy = np.concatenate([syn_num, syn_cat], axis=1)
    
    syn_pd = data_transformer.inverse_transform(syn_numpy)

    return syn_pd