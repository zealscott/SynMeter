import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings

from tqdm import tqdm
import time

from .vae_model import Model_VAE, Encoder_model, Decoder_model
from .utils_train import TabularDataset, my_collate
from lib.commons import read_csv
import pandas as pd
import os
from .data_transformer import DataTransformer

warnings.filterwarnings("ignore")


LR = 1e-3
WD = 0
D_TOKEN = 4
TOKEN_BIAS = True

N_HEAD = 1
FACTOR = 32
NUM_LAYERS = 2


def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss = (
        (X_num - Recon_X_num).pow(2).mean() if X_num is not None else torch.tensor(0.0)
    )
    device = mu_z.device
    ce_loss = torch.tensor(0.0).to(device)
    acc = torch.tensor(0.0).to(device)
    total_num = 0

    for idx, x_cat in enumerate(Recon_X_cat):
        if x_cat is not None:
            # whether X_cat values is out of range
            # this would happen only when evaluate the test data, which is not important
            n_class = x_cat.size(1)
            if X_cat[:, idx].max() < n_class:
                ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
            else:
                pass
            x_hat = x_cat.argmax(dim=-1)
        acc += (x_hat == X_cat[:, idx]).float().sum()
        total_num += x_hat.shape[0]

    if Recon_X_cat:
        ce_loss /= idx + 1
        acc /= total_num
    # loss = mse_loss + ce_loss

    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()

    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
    return mse_loss, ce_loss, loss_kld, acc


def train_vae(args, device):
    path_params = args["path_params"]
    model_params = args["model_params"]

    max_beta = model_params["max_beta"]
    min_beta = model_params["min_beta"]
    lambd = model_params["lambd"]

    # get the saved directory
    out_dir = os.path.dirname(path_params["out_model"])
    model_save_path = f"{out_dir}/vae_model.pt"
    encoder_save_path = f"{out_dir}/encoder.pt"
    decoder_save_path = f"{out_dir}/decoder.pt"

    # prepare data
    train_data_pd, meta_data, discrete_columns = read_csv(
        path_params["train_data"], path_params["meta_data"]
    )
    val_data_pd, _, _ = read_csv(path_params["val_data"], path_params["meta_data"])
    test_data_pd, _, _ = read_csv(path_params["test_data"], path_params["meta_data"])
    train_pd = pd.concat([train_data_pd, val_data_pd], ignore_index=True, sort=False)

    data_transformer = DataTransformer()
    num_train, cat_train = data_transformer.fit_transform(train_pd, discrete_columns)
    num_test, cat_test = data_transformer.transform(test_data_pd)
    d_numerical = data_transformer.get_num_dim()
    categories = data_transformer.get_cat_dim()

    if num_train is not None:
        num_train = torch.tensor(num_train, dtype=torch.float32)
        num_test = torch.tensor(num_test, dtype=torch.float32)
        num_test = num_test.float().to(device)
    if cat_train is not None:
        cat_train = torch.tensor(cat_train, dtype=torch.long)
        cat_test = torch.tensor(cat_test, dtype=torch.long)
        cat_test = cat_test.to(device)

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

    pre_encoder = Encoder_model(
        NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head=N_HEAD, factor=FACTOR
    ).to(device)
    pre_decoder = Decoder_model(
        NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head=N_HEAD, factor=FACTOR
    ).to(device)

    pre_encoder.eval()
    pre_decoder.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.95, patience=10, verbose=True
    )

    num_epochs = 4000
    best_train_loss = float("inf")

    current_lr = optimizer.param_groups[0]["lr"]
    patience = 0

    beta = max_beta
    start_time = time.time()
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

            batch_length = (
                batch_num.shape[0] if batch_num is not None else batch_cat.shape[0]
            )
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

        """
            Evaluation
        """
        model.eval()
        with torch.no_grad():
            Recon_X_num, Recon_X_cat, mu_z, std_z = model(num_test, cat_test)

            val_mse_loss, val_ce_loss, val_kl_loss, val_acc = compute_loss(
                num_test, cat_test, Recon_X_num, Recon_X_cat, mu_z, std_z
            )
            val_loss = val_mse_loss.item() * 0 + val_ce_loss.item()

            scheduler.step(val_loss)

        print(
            "epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Val MSE:{:.6f}, Val CE:{:.6f}, Train ACC:{:6f}, Val ACC:{:6f}".format(
                epoch,
                beta,
                num_loss,
                cat_loss,
                kl_loss,
                val_mse_loss.item(),
                val_ce_loss.item(),
                train_acc.item(),
                val_acc.item(),
            )
        )

    end_time = time.time()
    print("Training time: {:.4f} mins".format((end_time - start_time) / 60))

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

        np.save(f"{out_dir}/train_z.npy", train_z)
        
        print("Successfully save pretrained embeddings in disk {}".format(f"{out_dir}/train_z.npy"))

        # save the data transformer
        data_transformer.save(f"{out_dir}/data_transformer.pkl")
        
        print("Successfully save data transformer in disk {}".format(f"{out_dir}/data_transformer.pkl"))

