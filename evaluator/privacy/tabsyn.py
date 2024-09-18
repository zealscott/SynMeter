# adapt from https://github.com/sdv-dev/TVAE/tree/master
import torch
import os
from lib.info import *
from evaluator.privacy.eval_helper import sample_half_data
from evaluator.privacy.tabsyn_utils import tvae_trainer, diffusion_trainer, sample_from_tabsyn


def train_and_sample(config, data, cur_shadow_dir, cuda, n_syn_dataset):
    """
    Train and sample TabSyn given half of the data.
    Save the model and samples to the attack directory.

    Args:
        config (dict): Dictionary of arguments for training and sampling.
        data (tuple): Tuple containing all_data_pd, discrete_columns, meta_data, and dup_list.
        membership_info (list): List containing membership information.
        id (int): Identifier for the model and samples.
        cur_shadow_dir (str): Directory to save the model and samples.
        cuda (str): CUDA device identifier.

    Returns:
        None
    """
    shadow_data_pd, discrete_columns, meta_data = data
    device = torch.device("cuda:" + cuda)
    num_samples = config["sample_params"]["num_samples"]

    os.makedirs(cur_shadow_dir, exist_ok=True)

    print("start training TVAE...")
    data_transformer, pre_decoder, train_z = tvae_trainer(
        config, shadow_data_pd, discrete_columns, cur_shadow_dir, device
    )

    print("start training tabsyn...")
    diffusion_model = diffusion_trainer(config, train_z, cur_shadow_dir, device)

    print("start sampling...")
    for i in range(n_syn_dataset):
        sampled = sample_from_tabsyn(
            config, train_z, data_transformer, pre_decoder, num_samples, diffusion_model, device
        )
        sampled.to_csv(os.path.join(cur_shadow_dir, "sampled_{}.csv".format(i)), index=False)
