# adapt from https://github.com/sdv-dev/TVAE/tree/master
import torch
import os
from lib.info import *
from evaluator.privacy.eval_helper import sample_half_data
from synthesizer.ddpm.utils import make_dataloader
from synthesizer.ddpm.train_wrapper import Trainer
from synthesizer.ddpm.data_transformer import DataTransformer
from synthesizer.ddpm.modules import MLPDiffusion
from synthesizer.ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion


def tabddpm_trainer(args, data_pd, discrete_columns, meta_data, device):
    model_params = args["model_params"]
    path_params = args["path_params"]
    rtdl_params = {}
    rtdl_params["d_layers"] = model_params["d_layers"]
    rtdl_params["dropout"] = model_params["dropout"]

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

    trainer.run_loop()

    return trainer


def train_and_sample(config, data, cur_shadow_dir, cuda, n_syn_dataset):
    """
    Train and sample TabDDPM given half of the data.
    Save the model and samples to the attack directory.

    Args:
        args (dict): Dictionary of arguments for training and sampling.
        data (tuple): Tuple containing all_data_pd, discrete_columns, meta_data, and dup_list.
        membership_info (list): List containing membership information.
        id (int): Identifier for the model and samples.
        attack_dir (str): Directory to save the model and samples.
        cuda (str): CUDA device identifier.

    Returns:
        None
    """
    shadow_data_pd, discrete_columns, meta_data = data
    device = torch.device("cuda:" + cuda)
    # fit the model enough more to evaluate the privacy risk
    # args["model_params"]["steps"] = 20000
    # args["model_params"]["num_timesteps"] = 1000
    num_samples = len(shadow_data_pd)

    print("start training...")
    trainer = tabddpm_trainer(config, shadow_data_pd, discrete_columns, meta_data, device)

    # save the model
    os.makedirs(cur_shadow_dir, exist_ok=True)
    torch.save(trainer.diffusion, os.path.join(cur_shadow_dir, "model.pt"))

    print("start sampling...")
    for i in range(n_syn_dataset):
        data_transformer = trainer.diffusion.data_transformer
        empirical_class_dist = data_transformer.empirical_class_dist

        gen_x, gen_y = trainer.diffusion.sample_all(num_samples, batch_size=10000, y_dist=empirical_class_dist)

        sampled = data_transformer.inverse_transform(gen_x, gen_y)
        sampled.to_csv(os.path.join(cur_shadow_dir, "sampled_{}.csv".format(i)), index=False)
