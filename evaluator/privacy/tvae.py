# adapt from https://github.com/sdv-dev/TVAE/tree/master
from synthesizer.tvae import init_model
import torch
import os
from lib.info import *
from evaluator.privacy.eval_helper import sample_half_data


def train_and_sample(config, data, cur_shadow_dir, cuda, n_syn_dataset):
    """
    train TVAE using given config
    used when the best parameters are found and stored in `exp/`
    """
    shadow_data_pd, discrete_columns, meta_data = data
    device = torch.device("cuda:" + cuda)
    # we fit the model enough to evaluate the privacy risk
    config["model_params"]["epochs"] = 500
    num_samples = len(shadow_data_pd)

    model = init_model(config["model_params"], device)

    print("start training...")
    model.fit(shadow_data_pd, discrete_columns)

    # save the model
    os.makedirs(cur_shadow_dir, exist_ok=True)
    torch.save(model, os.path.join(cur_shadow_dir, "model.pt"))

    print("start sampling...")
    for i in range(n_syn_dataset):
        sampled = model.sample(num_samples)
        sampled.to_csv(os.path.join(cur_shadow_dir, "sampled_{}.csv".format(i)), index=False)
