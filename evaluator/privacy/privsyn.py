# adapt from https://github.com/sdv-dev/TVAE/tree/master
import torch
import os
from lib.info import *
from evaluator.privacy.eval_helper import sample_half_data
from synthesizer.syn.data_trasnformer import DataTransformer
from synthesizer.syn.data_loader import DataLoader
from synthesizer.syn.dpsyn import DPSyn
import pickle


def privsyn_trainer(args, data_pd, discrete_columns):
    model_params = args["model_params"]

    epsilon = model_params["epsilon"]
    delta = model_params["delta"]
    max_bins = model_params["max_bins"]
    update_iterations = model_params["update_iterations"]

    # budget allocation for DP
    ratio = model_params["ratio"] if "ratio" in model_params else None

    data_transformer = DataTransformer(max_bins)

    transformed_data = data_transformer.fit_transform(data_pd, discrete_columns)
    encode_mapping = data_transformer.get_mapping()

    # dataloader initialization
    data_loader = DataLoader()
    data_loader.load_data(private_data=transformed_data, encode_mapping=encode_mapping)

    synthesizer = DPSyn(data_loader, update_iterations, epsilon, delta, sensitivity=1, ratio=ratio)
    synthesizer.train()

    model = {}
    model["learned_privsyn"] = synthesizer
    model["data_transformer"] = data_transformer
    model["data_loader"] = data_loader

    return model


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
    # fit the model enough more to evaluate the privacy risk
    config["model_params"]["update_iterations"] = 100
    config["model_params"]["max_bins"] = 10
    config["model_params"]["epsilon"] = 100000000.0

    num_samples = len(shadow_data_pd)

    print("start training...")
    model = privsyn_trainer(config, shadow_data_pd, discrete_columns)

    # save the model
    os.makedirs(cur_shadow_dir, exist_ok=True)
    temp_path = os.path.join(cur_shadow_dir, "model.pt")
    pickle.dump(model, open(temp_path, "wb"))

    print("start sampling...")
    for i in range(n_syn_dataset):
        learned_privsyn = model["learned_privsyn"]
        data_transformer = model["data_transformer"]
        data_loader = model["data_loader"]

        # sample the same number of data as the real data
        syn_data = learned_privsyn.synthesize(num_records=num_samples)

        sampled = data_transformer.inverse_transform(syn_data)

        sampled.to_csv(os.path.join(cur_shadow_dir, "sampled_{}.csv".format(i)), index=False)
