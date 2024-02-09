"""
This script is used to tune the hyperparameters of the evualator (ML models) using REAL datasets
and store the best hyperparameters in a toml file
"""
from evaluator.utility.tab_transformer import tune_tab_transformer
from evaluator.utility.cat_boost import tune_catboost
from evaluator.utility.xgb import tune_xgb
from lib.commons import read_csv, preprocess, get_n_class
import argparse
from evaluator.utility import simple_evaluators
from lib.info import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="faults")
    parser.add_argument("--cuda", "-c", type=str, default="0")

    args = parser.parse_args()
    dataset = args.dataset

    train_data_path = ROOT_DIR + "/datasets/{0}/train.csv".format(dataset)
    val_data_path = ROOT_DIR + "/datasets/{0}/val.csv".format(dataset)
    meta_data_path = ROOT_DIR + "/datasets/{0}/{0}.json".format(dataset)

    # read data
    train_data_pd, meta_data, discrete_cols = read_csv(train_data_path, meta_data_path)
    val_data_pd, _, _ = read_csv(val_data_path, meta_data_path)
    n_class = get_n_class(meta_data_path)

    # preprocess data
    train_data, val_data, encodings = preprocess(train_data_pd, val_data_pd, meta_data, discrete_cols)

    task_type = meta_data["task"]

    # tune simple evaluator
    simple_evaluators.tune_lr(train_data, val_data, task_type, n_class, dataset, True)
    simple_evaluators.tune_mlp(train_data, val_data, task_type, n_class, dataset, True)
    simple_evaluators.tune_rf(train_data, val_data, task_type, n_class, dataset, True)
    simple_evaluators.tune_tree(train_data, val_data, task_type, n_class, dataset, True)
    simple_evaluators.tune_svm(train_data, val_data, task_type, n_class, dataset, True)

    tune_tab_transformer(train_data, val_data, task_type, n_class, dataset, True, "cuda:" + args.cuda)
    tune_catboost(train_data, val_data, task_type, n_class, dataset, True)
    tune_xgb(train_data, val_data, task_type, n_class, dataset, True)


if __name__ == "__main__":
    main()
