# adapt from https://github.com/sdv-dev/TVAE/tree/master
import torch
import os
from lib.info import *
from evaluator.privacy.eval_helper import sample_half_data
import pickle
from synthesizer.pgm.data_trasnformer import DataTransformer
from synthesizer.pgm import Dataset, reverse_data
from synthesizer.pgm.train_wrapper import MST_no_privacy, MST_private


def mst_trainer(args, data_pd, discrete_columns):
    model_params = args["model_params"]

    epsilon = model_params["epsilon"]
    delta = model_params["delta"]
    max_bins = model_params["max_bins"]
    bi_nums = model_params["bi_nums"]
    # tri_nums = model_params["tri_nums"]
    num_iters = model_params["num_iters"]
    cliques2 = model_params["2_cliques"]
    # cliques3 = model_params["3_cliques"]

    data_transformer = DataTransformer(max_bins)

    transformed_data, domain = data_transformer.fit_transform(data_pd, discrete_columns)

    data = Dataset.load(transformed_data, domain)

    # learned_pgm, supports = MST_no_privacy(
    #     data, epsilon, delta, [bi_nums, tri_nums], num_iters, cliques2, cliques3, device="cpu"
    # )

    learned_pgm, supports = MST_private(data, epsilon, delta, bi_nums, num_iters, cliques2, device="cpu")

    model = {}
    model["learned_pgm"] = learned_pgm
    model["data_transformer"] = data_transformer
    model["supports"] = supports

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
    config["model_params"]["num_iters"] = 5000
    config["model_params"]["max_bins"] = 5
    config["model_params"]["epsilon"] = 30000000.0
    config["model_params"]["delta"] = 1e-8

    num_samples = len(shadow_data_pd)

    print("start training...")
    model = mst_trainer(config, shadow_data_pd, discrete_columns)

    # save the model
    os.makedirs(cur_shadow_dir, exist_ok=True)
    temp_path = os.path.join(cur_shadow_dir, "model.pt")
    pickle.dump(model, open(temp_path, "wb"))

    print("start sampling...")
    for i in range(n_syn_dataset):
        learned_pgm = model["learned_pgm"]
        supports = model["supports"]
        data_transformer = model["data_transformer"]

        synth = learned_pgm.synthetic_data(rows=num_samples)
        syn_data = reverse_data(synth, supports)

        sampled = data_transformer.inverse_transform(syn_data.df)

        sampled.to_csv(os.path.join(cur_shadow_dir, "sampled_{}.csv".format(i)), index=False)
