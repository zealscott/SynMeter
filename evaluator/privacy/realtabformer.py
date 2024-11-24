from realtabformer import REaLTabFormer
import torch
import os
from lib.info import *
from evaluator.privacy.eval_helper import sample_half_data


def init_model(model_params):
    model = REaLTabFormer(
        model_type="tabular",
        gradient_accumulation_steps=model_params["gradient_accumulation_steps"],
        logging_steps=100,
        epochs=100,
    )

    return model


def train_and_sample(config, data, cur_shadow_dir, cuda, n_syn_dataset):
    """
    train TVAE using given config
    used when the best parameters are found and stored in `exp/`
    """
    shadow_data_pd, discrete_columns, meta_data = data
    # we fit the model enough to evaluate the privacy risk
    num_samples = len(shadow_data_pd)

    model = init_model(config["model_params"])

    print("start training...")
    model.experiment_id = "0"
    model.fit(shadow_data_pd, n_critic=0)

    # save the model
    model.save(cur_shadow_dir, os.path.join(cur_shadow_dir, "model.pt"))

    print("start sampling...")
    # sample from the trained model
    for i in range(n_syn_dataset):
        # sample from the trained model
        sampled = model.sample(num_samples)
        # remove space in column names and values with strip
        sampled.columns = sampled.columns.str.strip()
        sampled = sampled.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        sampled.to_csv(os.path.join(cur_shadow_dir, "sampled_{}.csv".format(i)), index=False)
