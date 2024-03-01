from lib.commons import *
import pandas as pd
import numpy as np
import itertools
import random
from lib.info import TUNED_PARAMS_PATH
from evaluator.utility.xgb import train_xgb
from evaluator.utility.eval_helper import query_evaluation, ml_evaluation
from evaluator.fidelity.eval_helper import fidelity_evaluation
from evaluator.utility.util import split_data_stratify


def fidelity_tuner(config,seed):
    """
    calculate the fidelity of the synthetic data
    use mean error of all fidelity metrics
    """
    results = fidelity_evaluation(config, seed, tune=True)
    error = []
    for metric, value in results.items():
        error.append(value)
    return sum(error) / len(error)


def utility_tuner(config, dataset, cuda, seed):
    """
    calculate the utility of the synthetic data
    use mean error of affinity and query error
    """
    ########## ML affinity ##########
    ml_syn_results, ml_real_results = ml_evaluation(config, dataset, cuda, seed, tune=True)
    # calculate the affinity
    affinity = []
    for evaluator, res in ml_syn_results.items():
        metric = "rmse" if "rmse" in res else "f1"
        syn_ans = res[metric]
        real_ans = ml_real_results[evaluator][metric]
        affinity.append(abs(syn_ans - real_ans) / real_ans)
    affinity = sum(affinity) / len(affinity)

    ########## range query error ##########
    query_res = query_evaluation(config, query_res={}, tune=True)
    query_error = []
    for metric, value in query_res.items():
        query_error.append(value)
    query_error = sum(query_error) / len(query_error)

    return affinity, query_error


# for naive
def select_two_clique(columns, n=30):
    # get all 2 cliques
    cliques = list(itertools.combinations(columns, 2))
    # filter with label
    label_cliques = [c for c in cliques if "label" in c]
    n_left = min(max(n - len(label_cliques), 0), len(cliques))
    # randomly select n_left cliques
    if n_left == 0:
        print("label clique is not included ALL in the cliques")
    selected_cliques = label_cliques + random.sample(cliques, n_left)
    return selected_cliques


def select_three_clique(columns, n=10):
    # get all 3 cliques
    cliques = list(itertools.combinations(columns, 3))
    return random.sample(cliques, n)
