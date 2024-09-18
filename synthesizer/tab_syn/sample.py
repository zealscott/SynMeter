import torch

import warnings
import time

from .model import MLPDiffusion, Model
from .latent_utils import get_input_generate, recover_data
from .diffusion_utils import sample

warnings.filterwarnings("ignore")


def sample_from_tabsyn(args, n_samples, device):

    train_z, out_model_path, data_transformer, pre_decoder, token_dim = (
        get_input_generate(args)
    )
    in_dim = train_z.shape[1]

    mean, std = train_z.mean(0), train_z.std(0)

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    pre_decoder = pre_decoder.to(device)

    model = Model(denoise_fn=denoise_fn, hid_dim=train_z.shape[1]).to(device)

    model.load_state_dict(torch.load(out_model_path))

    """
        Generating samples    
    """
    start_time = time.time()

    x_next = sample(model.denoise_fn_D, n_samples, in_dim, device=device)
    
    # x_next = x_next * 2 + mean.to(device)
    ###############################
    # Yuntao: I think the above line is wrong, it should be:
    x_next = x_next * (std.to(device) + 1e-8) + mean.to(device)
    ###############################

    syn_data = x_next.float().cpu().numpy()

    syn_df = recover_data(syn_data, data_transformer, pre_decoder, token_dim,device)

    end_time = time.time()
    print("Time:", end_time - start_time)

    return syn_df
