"""
This script is used to evaluate the diversity of the synthetic data
"""

import os
import argparse
from lib.commons import load_config
from evaluator.diversity.eval_helper import save_diversity_results, diversity_evaluation
from lib.info import *





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="tabddpm_ori")
    parser.add_argument("--dataset", "-d", type=str, default="faults")
    parser.add_argument("--cuda", "-c", type=str, default="0")
    parser.add_argument("--use_train", "-u", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    use_train = args.use_train

    print("Evalute diversity performance for dataset {0} with algorithm {1}".format(args.dataset, args.model))

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

    diversity_results = {}
    for i in range(N_EXPS):
        print("Evaluate diversity experiment {0}/{1}".format(i + 1, N_EXPS))
        seed = i
        synthesizer.sample(config, n_samples, seed)
        # evalute diversity
        diversity_results = diversity_evaluation(config, diversity_results,use_train)

    # save the result
    save_path = config["path_params"]["diversity_result"]
    save_path = save_path + "_train" if use_train else save_path + "_test"
    save_diversity_results(diversity_results, save_path)


if __name__ == "__main__":
    main()
