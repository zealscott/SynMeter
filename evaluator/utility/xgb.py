from xgboost import XGBClassifier, XGBRegressor
import numpy as np
import optuna
from lib.commons import dump_config, cal_metrics
from evaluator.utility.util import callback, get_score, missing_class_corrector
import os
from lib.info import TUNED_PARAMS_PATH


def train_xgb(params, train_data, test_data, task_type, n_class):
    """
    train xgboost with given parameters
    """
    x_train, y_train = train_data

    if task_type == "regression":
        model = XGBRegressor(
            **params,
            objective="reg:squarederror",
        )
    else:
        model = XGBClassifier(
            **params,
            objective="binary:logistic" if task_type == "binary_classification" else "multi:softmax",
        )

    const_pred, unique_labels = missing_class_corrector(train_data, test_data, task_type)
    if const_pred is None:
        model.fit(x_train, y_train, verbose=100)

    score = get_score(model, test_data, task_type, n_class, const_pred, unique_labels)
    return model, score


def tune_xgb(train_data, val_data, task_type, n_class, dataset=None, store_best_params=True):
    """
    tune xgboost with optuna
    when store_best_params is True, we save the best parameters
    otherwise, we return the best model and its score on test data
    note: we only use train and val data for tuning, never use test data
    """

    def xgb_objective(trial):
        # configure the model for this trial
        eta = trial.suggest_float("eta", 0.01, 0.2, log=False)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        gamma = trial.suggest_float("gamma", 0, 1)

        xgb_params = {
            "eta": eta,
            "min_child_weight": min_child_weight,
            "max_depth": max_depth,
            "gamma": gamma,
        }

        # train model with train data, test on val data
        model, res = train_xgb(xgb_params, train_data, val_data, task_type, n_class)
        res = res["rmse"] if task_type == "regression" else res["f1"]

        # save the model
        trial.set_user_attr("best_model", model)
        return res

    study = optuna.create_study(
        direction="maximize" if task_type != "regression" else "minimize",
        sampler=optuna.samplers.TPESampler(seed=0),
    )

    study.optimize(xgb_objective, n_trials=100, show_progress_bar=True, callbacks=[callback])
    if store_best_params:
        # save the best params
        # used for tune best parameters for real data
        best_config = study.best_trial.params
        parent_path = TUNED_PARAMS_PATH + "/evaluators/xgboost"
        out_path = parent_path + "/{0}.toml".format(dataset)
        os.makedirs(parent_path, exist_ok=True)
        dump_config(best_config, out_path)

    # get the test score of the best model
    best_model = study.user_attrs["best_model"]

    const_pred, unique_labels = missing_class_corrector(train_data, val_data, task_type)
    score = get_score(best_model, val_data, task_type, n_class, const_pred, unique_labels)
    res = score["rmse"] if task_type == "regression" else score["f1"]
    return best_model, res
