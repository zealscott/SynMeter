import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from copy import deepcopy
from .gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from .utils import update_ema, make_dataloader
from .modules import MLPDiffusion
from lib.commons import read_csv
from .data_transformer import DataTransformer


class Trainer:
    def __init__(self, diffusion, train_iter, lr, weight_decay, steps, device):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_iter = train_iter
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=["step", "mloss", "gloss", "loss"])
        self.log_every = 100
        self.print_every = 1000
        self.ema_every = 1000

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict=None):
        x = x.to(self.device)
        if out_dict:
            for k in out_dict:
                out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()
        loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)
        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss

    def run_loop(self):
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        while step < self.steps:
            iter_data = next(self.train_iter)
            if len(iter_data) == 1:
                # not conditional on y
                x = iter_data[0]
                batch_loss_multi, batch_loss_gauss = self._run_step(x)
            else:
                x, out_dict = iter_data
                out_dict = {"y": out_dict}
                batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if (step + 1) % self.print_every == 0:
                    print(
                        f"Step {(step + 1)}/{self.steps} MLoss: {mloss:.4f} GLoss: {gloss:.4f} Sum: {(mloss + gloss):.4f}"
                    )
                self.loss_history.loc[len(self.loss_history)] = [step + 1, mloss, gloss, mloss + gloss]
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

            step += 1

    def audit_run_loop(self, interval=10, store_dir = None, n_sample = 100):
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        while step < self.steps:
            iter_data = next(self.train_iter)
            if len(iter_data) == 1:
                # not conditional on y
                x = iter_data[0]
                batch_loss_multi, batch_loss_gauss = self._run_step(x)
            else:
                x, out_dict = iter_data
                out_dict = {"y": out_dict}
                batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if (step + 1) % self.print_every == 0:
                    print(
                        f"Step {(step + 1)}/{self.steps} MLoss: {mloss:.4f} GLoss: {gloss:.4f} Sum: {(mloss + gloss):.4f}"
                    )
                self.loss_history.loc[len(self.loss_history)] = [step + 1, mloss, gloss, mloss + gloss]
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0
            
            ### audit the training process ###
            if (step) % interval == 0 or step ==0:
                # sample data
                data_transformer = self.diffusion.data_transformer

                # sample the same number of data as the real data
                empirical_class_dist = data_transformer.empirical_class_dist

                gen_x, gen_y = self.diffusion.sample_all(n_sample, batch_size=10000, y_dist=empirical_class_dist)

                sampled = data_transformer.inverse_transform(gen_x, gen_y)
                import os
                os.makedirs(store_dir, exist_ok=True)
                sample_path = os.path.join(store_dir, 'gen_data_{}.csv'.format(step))
                sampled.to_csv(sample_path, index=False)
                print("save the sample data at {}".format(sample_path))

            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

            step += 1
    

def train_wrapper_tab_ddpm(args, device, tune=False):
    model_params = args["model_params"]
    path_params = args["path_params"]
    rtdl_params = {}
    rtdl_params["d_layers"] = model_params["d_layers"]
    rtdl_params["dropout"] = model_params["dropout"]

    # prepare data
    train_data_pd, meta_data, discrete_columns = read_csv(path_params["train_data"], path_params["meta_data"])
    val_data_pd, _, _ = read_csv(path_params["val_data"], path_params["meta_data"])
    if tune:
        data_pd = train_data_pd
    else:
        # combine train and val data
        data_pd = pd.concat([train_data_pd, val_data_pd], ignore_index=True, sort=False)

    # conditional on y when classification
    y_cond = False if meta_data["task"] == "regression" else True

    # quantial transform and one-hot encoding
    # if classifcation, condtional on y
    data_transformer = DataTransformer(y_cond)
    transformed_data, transformed_label = data_transformer.fit_transform(data_pd, discrete_columns)
    # convert to torch tensor
    transformed_data = torch.tensor(transformed_data, dtype=torch.float32)
    # prepare data loader
    transformed_label = torch.tensor(transformed_label, dtype=torch.float32)
    train_loader = make_dataloader(
        transformed_data,
        transformed_label,
        batch_size=model_params["batch_size"],
        shuffle=True,
    )

    input_dim = data_transformer.get_dim()
    n_label = data_transformer.get_label_dim()

    model = MLPDiffusion(d_in=input_dim, num_classes=n_label, is_y_cond=y_cond, rtdl_params=rtdl_params)

    diffusion = GaussianMultinomialDiffusion(
        num_classes=data_transformer.get_cat_dim(),
        num_numerical_features=data_transformer.get_num_dim(),
        denoise_fn=model,
        num_timesteps=model_params["num_timesteps"],
        data_transformer=data_transformer,
        device=device,
    )

    diffusion.to(device)
    diffusion.train()

    # prepare training parameters
    lr = model_params["lr"]
    weight_decay = model_params["weight_decay"]
    steps = model_params["steps"]

    trainer = Trainer(diffusion, train_loader, lr, weight_decay, steps, device)
    return trainer, diffusion
