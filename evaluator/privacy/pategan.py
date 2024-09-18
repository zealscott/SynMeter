import torch
import os
from lib.info import *
from evaluator.privacy.eval_helper import sample_half_data
from synthcity.plugins import Plugins
from synthcity.utils.serialization import save_to_file, load_from_file
from synthcity.plugins.core.dataloader import GenericDataLoader


def train_and_sample(config, data, cur_shadow_dir, cuda, n_syn_dataset):
    """
    train CTGAN using given config
    used when the best parameters are found and stored in `exp/`
    """
    shadow_data_pd, discrete_columns, meta_data = data
    device = torch.device("cuda:" + cuda)
    # we fit the model enough to evaluate the privacy risk
    model_params = config["model_params"]
    num_samples = len(shadow_data_pd)

    loader = GenericDataLoader(shadow_data_pd)

    model = Plugins().get("pategan", **model_params, device=device)

    print("start training...")
    model.fit(loader)

    # save the model
    os.makedirs(cur_shadow_dir, exist_ok=True)
    save_to_file(os.path.join(cur_shadow_dir, "model.pt"), model)

    print("start sampling...")
    # sample from the trained model
    for i in range(n_syn_dataset):
        # sample from the trained model
        sampled = model.generate(num_samples).dataframe()
        sampled.to_csv(os.path.join(cur_shadow_dir, "sampled_{}.csv".format(i)), index=False)
