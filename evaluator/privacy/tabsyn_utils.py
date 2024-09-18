import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
from tqdm import tqdm

import torch
from synthesizer.tab_syn.data_transformer import DataTransformer
from synthesizer.tab_syn.utils_train import TabularDataset, my_collate
from synthesizer.tab_syn.vae_model import Model_VAE, Encoder_model, Decoder_model
from synthesizer.tab_syn.vae_main import compute_loss
from synthesizer.tab_syn.model import MLPDiffusion, Model
from synthesizer.tab_syn.diffusion_utils import sample
from synthesizer.tab_syn.latent_utils import recover_data


warnings.filterwarnings("ignore")


LR = 1e-3
WD = 0
D_TOKEN = 4
TOKEN_BIAS = True

N_HEAD = 1
FACTOR = 32
NUM_LAYERS = 2


def tvae_trainer(args, data_pd, discrete_columns, cur_shadow_dir, device):
    model_params = args["model_params"]

    max_beta = model_params["max_beta"]
    min_beta = model_params["min_beta"]
    lambd = model_params["lambd"]

    model_save_path = f"{cur_shadow_dir}/vae_model.pt"
    encoder_save_path = f"{cur_shadow_dir}/encoder.pt"
    decoder_save_path = f"{cur_shadow_dir}/decoder.pt"

    data_transformer = DataTransformer()
    num_train, cat_train = data_transformer.fit_transform(data_pd, discrete_columns)
    d_numerical = data_transformer.get_num_dim()
    categories = data_transformer.get_cat_dim()

    if num_train is not None:
        num_train = torch.tensor(num_train, dtype=torch.float32)
    if cat_train is not None:
        cat_train = torch.tensor(cat_train, dtype=torch.long)

    train_data = TabularDataset(num_train, cat_train)

    batch_size = 4096
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=my_collate,
    )

    model = Model_VAE(
        NUM_LAYERS,
        d_numerical,
        categories,
        D_TOKEN,
        n_head=N_HEAD,
        factor=FACTOR,
        bias=True,
    )
    model = model.to(device)

    pre_encoder = Encoder_model(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head=N_HEAD, factor=FACTOR).to(device)
    pre_decoder = Decoder_model(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head=N_HEAD, factor=FACTOR).to(device)

    pre_encoder.eval()
    pre_decoder.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.95, patience=10, verbose=True)

    num_epochs = 4000
    best_train_loss = float("inf")

    current_lr = optimizer.param_groups[0]["lr"]
    patience = 0

    beta = max_beta
    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0
        curr_loss_kl = 0.0

        curr_count = 0

        for batch_num, batch_cat in pbar:
            model.train()
            optimizer.zero_grad()

            batch_num = batch_num.to(device) if batch_num is not None else None
            batch_cat = batch_cat.to(device) if batch_cat is not None else None

            Recon_X_num, Recon_X_cat, mu_z, std_z = model(batch_num, batch_cat)

            loss_mse, loss_ce, loss_kld, train_acc = compute_loss(
                batch_num, batch_cat, Recon_X_num, Recon_X_cat, mu_z, std_z
            )

            loss = loss_mse + loss_ce + beta * loss_kld
            loss.backward()
            optimizer.step()

            batch_length = batch_num.shape[0] if batch_num is not None else batch_cat.shape[0]
            curr_count += batch_length
            curr_loss_multi += loss_ce.item() * batch_length
            curr_loss_gauss += loss_mse.item() * batch_length
            curr_loss_kl += loss_kld.item() * batch_length

        num_loss = curr_loss_gauss / curr_count
        cat_loss = curr_loss_multi / curr_count
        kl_loss = curr_loss_kl / curr_count

        train_loss = num_loss + cat_loss
        scheduler.step(train_loss)

        new_lr = optimizer.param_groups[0]["lr"]

        if new_lr != current_lr:
            current_lr = new_lr
            print(f"Learning rate updated: {current_lr}")

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            patience = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience += 1
            if patience == 10:
                if beta > min_beta:
                    beta = beta * lambd

        print(
            "epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}".format(
                epoch,
                beta,
                num_loss,
                cat_loss,
                kl_loss,
            )
        )

    # Saving latent embeddings
    with torch.no_grad():
        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model)

        torch.save(pre_encoder.state_dict(), encoder_save_path)
        torch.save(pre_decoder.state_dict(), decoder_save_path)

        num_train = num_train.to(device) if num_train is not None else None
        cat_train = cat_train.to(device) if cat_train is not None else None

        print("Successfully load and save the model {}!".format(model_save_path))

        train_z = pre_encoder(num_train, cat_train).detach().cpu().numpy()

        np.save(f"{cur_shadow_dir}/train_z.npy", train_z)

        print("Successfully save pretrained embeddings in disk {}".format(f"{cur_shadow_dir}/train_z.npy"))

        # save the data transformer
        data_transformer.save(f"{cur_shadow_dir}/data_transformer.pkl")

        print("Successfully save data transformer in disk {}".format(f"{cur_shadow_dir}/data_transformer.pkl"))

    return data_transformer, pre_decoder, train_z


def diffusion_trainer(args, train_z, cur_shadow_dir, device):
    """
    train diffusion models with trained VAE embeddings
    """
    out_model_path = f"{cur_shadow_dir}/tabsyn.pt"
    model_params = args["model_params"]
    batch_size = model_params["batch_size"]
    num_epochs = model_params["num_epochs"]
    lr = model_params["lr"]

    train_z = torch.tensor(train_z).float()
    train_z = train_z[:, 1:, :]  # remove the first token
    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    # flatten the input
    train_z = train_z.view(B, in_dim)

    mean, std = train_z.mean(0), train_z.std(0)

    # Normalize to standard normal, why divide by 2?
    train_z = (train_z - mean) / 2
    train_data = train_z

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    print(denoise_fn)

    num_params = sum(p.numel() for p in denoise_fn.parameters())
    print("the number of parameters", num_params)

    model = Model(denoise_fn=denoise_fn, hid_dim=train_z.shape[1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=20, verbose=True)

    model.train()

    best_loss = float("inf")
    patience = 0
    for epoch in range(num_epochs):

        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        batch_loss = 0.0
        len_input = 0
        for batch in pbar:
            inputs = batch.float().to(device)
            loss = model(inputs)

            loss = loss.mean()

            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": loss.item()})

        curr_loss = batch_loss / len_input
        scheduler.step(curr_loss)

        if curr_loss < best_loss:
            best_loss = loss.item()
            patience = 0
            torch.save(model.state_dict(), out_model_path)
        else:
            patience += 1
            if patience == 500:
                print("Early stopping")
                break
    return model


def sample_from_tabsyn(args, train_z, data_transformer, pre_decoder, n_samples, diffusion_model, device):
    train_z = torch.tensor(train_z).float()
    train_z = train_z[:, 1:, :]
    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    train_z = train_z.view(B, in_dim)

    in_dim = train_z.shape[1]

    mean = train_z.mean(0)

    x_next = sample(diffusion_model.denoise_fn_D, n_samples, in_dim, device=device)
    x_next = x_next * 2 + mean.to(device)

    syn_data = x_next.float().cpu().numpy()

    syn_df = recover_data(syn_data, data_transformer, pre_decoder, token_dim, device)

    return syn_df
