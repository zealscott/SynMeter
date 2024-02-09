# adapt from https://github.com/sdv-dev/TVAE/tree/master
from synthesizer.tvae import init_model
import torch
import os
from lib.info import *
from evaluator.privacy.eval_helper import sample_half_data


def train_and_sample(args, data, membership_info, id, attack_dir, cuda):
    """
    train TVAE using given args
    used when the best parameters are found and stored in `exp/`
    """
    all_data_pd, discrete_columns, meta_data, dup_list = data
    device = torch.device("cuda:" + cuda)
    # we fit the model enough to evaluate the privacy risk
    args["model_params"]["epochs"] = 500
    num_samples = args["sample_params"]["num_samples"]

    train_index = sample_half_data(all_data_pd, dup_list, membership_info)
    train_data_pd = all_data_pd.iloc[train_index]

    model = init_model(args["model_params"], device)

    print("start training...")
    model.fit(train_data_pd, discrete_columns)

    # save the model
    os.makedirs(attack_dir, exist_ok=True)
    torch.save(model, os.path.join(attack_dir, "model_{}.pt".format(id)))

    print("start sampling...")
    # sample from the trained model
    sampled = model.sample(num_samples)
    sampled.to_csv(os.path.join(attack_dir, "sampled_{}.csv".format(id)), index=False)
