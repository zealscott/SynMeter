import torch
import os
from lib.info import *
from evaluator.privacy.eval_helper import sample_half_data
from synthesizer.tablediff.modules import TableDiffusion_Synthesiser


def train_and_sample(config, data, cur_shadow_dir, cuda, n_syn_dataset):
    """
    train CTGAN using given config
    used when the best parameters are found and stored in `exp/`
    """
    shadow_data_pd, discrete_columns, meta_data = data
    device = torch.device("cuda:" + cuda)
    # we fit the model enough to evaluate the privacy risk
    batch_size = config["model_params"]["batch_size"]
    lr = config["model_params"]["lr"]
    dim = config["model_params"]["d_layers"]
    diffusion_steps = config["model_params"]["diffusion_steps"]
    predict_noise = config["model_params"]["predict_noise"]
    epsilon_target = config["model_params"]["epsilon_target"]
    epoch_target = config["model_params"]["epoch_target"]

    num_samples = len(shadow_data_pd)

    model = TableDiffusion_Synthesiser(
        batch_size=batch_size,
        lr=lr,
        dims=dim,
        diffusion_steps=diffusion_steps,
        predict_noise=predict_noise,
        epsilon_target=epsilon_target,
        epoch_target=epoch_target,
        device=device,
    )

    print("start training...")
    model.fit(df=shadow_data_pd, discrete_columns=discrete_columns)

    # save the model
    os.makedirs(cur_shadow_dir, exist_ok=True)
    torch.save(model, os.path.join(cur_shadow_dir, "model.pt"))

    print("start sampling...")
    for i in range(n_syn_dataset):
        sampled = model.sample(num_samples)
        sampled.to_csv(os.path.join(cur_shadow_dir, "sampled_{}.csv".format(i)), index=False)
