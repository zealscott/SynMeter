# adapt from https://github.com/sdv-dev/TVAE/tree/master
from synthesizer.great import init_model
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
    # we fit the model enough to evaluate the privacy risk
    args["model_params"]["epochs"] = 300
    num_samples = args["sample_params"]["num_samples"]

    train_index = sample_half_data(all_data_pd, dup_list, membership_info)
    train_data_pd = all_data_pd.iloc[train_index]
    
    model = init_model(args["model_params"], saved_dir= attack_dir)

    print("start training...")
    model.fit(train_data_pd, discrete_columns)

    # save the model
    model.save(attack_dir, os.path.join(attack_dir, "model_{}.pt".format(id)))

    print("start sampling...")
    # sample from the trained model
    sampled = model.sample(
        num_samples, k=100, device="cuda:0", temperature=args["model_params"]["temperature"]
    )
    
    # remove space in column names and values with strip
    sampled.columns = sampled.columns.str.strip()
    sampled = sampled.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    sampled.to_csv(os.path.join(attack_dir, "sampled_{}.csv".format(id)), index=False)
