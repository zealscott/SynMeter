import os
import torch

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
import time

from tqdm import tqdm
from .model import MLPDiffusion, Model
from .latent_utils import get_input_train

warnings.filterwarnings("ignore")


def train_diffusion(args, device):
    """
    train diffusion models with trained VAE embeddings
    """
    out_model_path = args["path_params"]["out_model"]
    model_params = args["model_params"]
    batch_size = model_params["batch_size"]
    num_epochs = model_params["num_epochs"]
    lr = model_params["lr"]

    train_z, out_dir = get_input_train(args)

    in_dim = train_z.shape[1]

    mean, std = train_z.mean(0), train_z.std(0)

    # Normalize to standard normal, why divide by 2?
    # train_z = (train_z - mean) / 2
    ###############################
    # Yuntao 2/18/2024: I think the above line is wrong, it should be:
    train_z = (train_z - mean) / (std + 1e-8)
    ###############################
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
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.9, patience=20, verbose=True
    )

    model.train()

    best_loss = float("inf")
    patience = 0
    start_time = time.time()
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

    end_time = time.time()
    print("Time: ", end_time - start_time)
