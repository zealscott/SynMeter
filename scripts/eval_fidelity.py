"""
This script is used to evaluate the fidelity of the synthetic data
column-wise: marginal error for discrete columns (range [0,1])
             wasserstein distance for continuous columns (range [0,inf])
pairwise: pearson correlation score (range [0,1])
          contigency error (range [0,1])
          correlation error (range [0,1])
"""
import os
import argparse
from lib.commons import load_config
from evaluator.fidelity.eval_helper import (
    save_fidelity_results,
    add_fidelity_results,
    fidelity_evaluation,
)
from lib.info import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="mst")
    parser.add_argument("--dataset", "-d", type=str, default="news")
    parser.add_argument("--type", "-t", type=str, default="test")
    parser.add_argument("--seed", "-s", type=int, default=0)

    args = parser.parse_args()

    print("Evalute fidelity for dataset {0} with algorithm {1}".format(args.dataset, args.model))
    # load template config
    model_config = "exp/{0}/{1}/config.toml".format(args.dataset, args.model)
    config = load_config(os.path.join(ROOT_DIR, model_config))

    seed = args.seed
    n_samples = config["sample_params"]["num_samples"]

    # dynamically import model interface
    synthesizer = __import__("synthesizer." + args.model, fromlist=[args.model])

    model_path = config["path_params"]["out_model"]
    if not os.path.exists(model_path):
        raise ValueError("Please train the synthesizer first (script/train_synthesizer.py)")

    fidelity_res = {}
    for i in range(N_EXPS):
        print("Evaluate fidelity {0}/{1}".format(i + 1, N_EXPS))
        seed = i
        synthesizer.sample(config, n_samples, seed)
        # evalute with statistical metrics
        cur_res = fidelity_evaluation(config,seed,eval_type=args.type)
        fidelity_res = add_fidelity_results(cur_res, fidelity_res)

    if args.type == "test":
        # save the result
        save_fidelity_results(fidelity_res, config["path_params"]["fidelity_result"])
    else:
        save_fidelity_results(fidelity_res, config["path_params"]["fidelity_train_result"])


if __name__ == "__main__":
    main()
