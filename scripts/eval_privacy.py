"""
This script is used to evaluate the attack of the synthetic data
"""

import os
import argparse
from lib.commons import load_config
from evaluator.privacy.eval_helper import privacy_evaluation, save_privacy_results
from lib.info import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="tablediffusion_eps_1")
    parser.add_argument("--dataset", "-d", type=str, default="bean")
    parser.add_argument("--n_gen", "-n", type=int, default=40)
    parser.add_argument("--cuda", "-c", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    print(
        "Evalute attack performance for dataset {0} with algorithm {1}".format(
            args.dataset, args.model
        )
    )

    # load template config
    model_config = "exp/{0}/{1}/config.toml".format(args.dataset, args.model)
    config = load_config(os.path.join(ROOT_DIR, model_config))

    ans = privacy_evaluation(config, args.n_gen, args.model, args.dataset, args.cuda)

    save_privacy_results(ans, config["path_params"]["privacy_result"])


if __name__ == "__main__":
    main()
