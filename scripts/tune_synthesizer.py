"""
Tune pipeline for synthesizer
Each synthesizer has its own tune function
"""

import os
import argparse
from lib.commons import load_config, dump_config
from lib.info import ROOT_DIR, TUNED_PARAMS_PATH


def update_config(config, model, dataset):
    """add model configure and dataset configure to args"""
    config["path_params"]["meta_data"] = "datasets/{0}/{0}.json".format(dataset)
    config["path_params"]["train_data"] = "datasets/{0}/train.csv".format(dataset)
    config["path_params"]["val_data"] = "datasets/{0}/val.csv".format(dataset)
    config["path_params"]["test_data"] = "datasets/{0}/test.csv".format(dataset)
    config["path_params"]["raw_data"] = "datasets/{0}/{0}.csv".format(dataset)

    config["path_params"]["loss_record"] = "exp/{0}/{1}/loss.csv".format(dataset, model)
    config["path_params"]["out_model"] = "exp/{0}/{1}/{1}.pt".format(dataset, model)
    config["path_params"]["out_data"] = "exp/{0}/{1}/{0}.csv".format(dataset, model)

    config["path_params"]["fidelity_result"] = "exp/{0}/{1}/fidelity_result.json".format(dataset, model)
    config["path_params"]["privacy_result"] = "exp/{0}/{1}/privacy_result.json".format(dataset, model)
    config["path_params"]["fidelity_train_result"] = "exp/{0}/{1}/fidelity_train_result.json".format(dataset, model)
    config["path_params"]["utility_result"] = "exp/{0}/{1}/utility_result.json".format(dataset, model)

    # set the absolute path
    for key, value in config["path_params"].items():
        config["path_params"][key] = os.path.join(ROOT_DIR, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="privsyn_eps_1")
    parser.add_argument("--dataset", "-d", type=str, default="obesity")
    parser.add_argument("--cuda", "-c", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # load model config
    model_config = "base_config/{0}.toml".format(args.model)
    config = load_config(os.path.join(TUNED_PARAMS_PATH, model_config))

    # modify config according to dataset and model
    update_config(config, args.model, args.dataset)

    seed = args.seed

    # dynamically import model interface
    synthesizer = __import__("synthesizer." + args.model, fromlist=[args.model])

    print("Tuning {0} on {1} with seed {2}".format(args.model, args.dataset, seed))
    tuned_config = synthesizer.tune(config, args.cuda, args.dataset, seed=seed)

    # save tuned parameters
    dump_path = os.path.join(TUNED_PARAMS_PATH, "{0}/{1}/config.toml".format(args.dataset, args.model))
    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    dump_config(tuned_config, dump_path)


if __name__ == "__main__":
    main()
