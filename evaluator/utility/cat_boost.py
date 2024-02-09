from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import f1_score, r2_score, roc_auc_score, mean_squared_error
import numpy as np
import optuna
from lib.commons import dump_config, cal_metrics
from evaluator.utility.util import callback, get_score, missing_class_corrector
import os
from lib.info import TUNED_PARAMS_PATH
from catboost import Pool

def train_catboost(params, train_data, test_data, task_type, n_class):
    x_train, y_train = train_data
    if task_type == "regression":
        model = CatBoostRegressor(
            **params,
            eval_metric="RMSE",  # R2 is not supported on GPU
            thread_count=-1,
        )
    else:
        model = CatBoostClassifier(
            **params,
            loss_function="MultiClass" if task_type == "multiclass_classification" else "Logloss",
            eval_metric="TotalF1" if task_type == "multiclass_classification" else "F1",
            thread_count=-1,
        )

    # train_pool = Pool(data=x_train, label=y_train)

    const_pred, unique_labels = missing_class_corrector(train_data, test_data, task_type)
    if const_pred is None:
        model.fit(x_train, y_train, verbose=1000)
        # model.fit(train_pool, verbose=1000)

    score = get_score(model, test_data, task_type, n_class, const_pred, unique_labels)
    return model, score


def tune_catboost(train_data, val_data, task_type, n_class, dataset=None, store_best_params=True):
    """
    tune catboost with optuna
    when store_best_params is True, we save the best parameters
    otherwise, we return the best model and its score on test data
    note: we only use train and val data for tuning, never use test data
    """

    def catboost_objective(trial):
        # configure the model for this trial
        lr = trial.suggest_float("lr", 1e-5, 1, log=True)
        l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1, 10, log=True)
        depth = trial.suggest_int("depth", 3, 10)
        bagging_temperature = trial.suggest_float("bagging_temperature", 0, 1)
        leaf_estimation_iterations = trial.suggest_int("leaf_estimation_iterations", 1, 10)
        # predefined params
        iterations = 3000
        early_stopping_rounds = 50
        od_pval = 0.001

        # get catboost configures
        catboost_params = {}
        catboost_params["learning_rate"] = lr
        catboost_params["l2_leaf_reg"] = l2_leaf_reg
        catboost_params["depth"] = depth
        catboost_params["bagging_temperature"] = bagging_temperature
        catboost_params["leaf_estimation_iterations"] = leaf_estimation_iterations
        catboost_params["iterations"] = iterations
        catboost_params["early_stopping_rounds"] = early_stopping_rounds
        catboost_params["od_pval"] = od_pval

        trial.set_user_attr("config", catboost_params)

        # train model with train data, test on val data
        model, res = train_catboost(catboost_params, train_data, val_data, task_type, n_class)
        res = res["rmse"] if task_type == "regression" else res["f1"]
        # save the model
        trial.set_user_attr("best_model", model)
        return res

    study = optuna.create_study(
        direction="maximize" if task_type != "regression" else "minimize",
        sampler=optuna.samplers.TPESampler(seed=0),
    )

    study.optimize(catboost_objective, n_trials=100, show_progress_bar=True, callbacks=[callback])
    if store_best_params:
        # save the best params
        # used for tune best parameters for real data
        best_config = study.best_trial.user_attrs["config"]
        parent_path = TUNED_PARAMS_PATH + "/evaluators/cat_boost"
        out_path = parent_path + "/{0}.toml".format(dataset)
        os.makedirs(parent_path, exist_ok=True)
        dump_config(best_config, out_path)
    # get the test score of the best model
    best_model = study.user_attrs["best_model"]

    const_pred, unique_labels = missing_class_corrector(train_data, val_data, task_type)
    score = get_score(best_model, val_data, task_type, n_class, const_pred, unique_labels)
    res = score["rmse"] if task_type == "regression" else score["f1"]
    return best_model, res
