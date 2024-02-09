import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from lib.commons import dump_config, cal_metrics
from evaluator.utility.util import callback
from tqdm.contrib.logging import logging_redirect_tqdm
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
import optuna
import os
from lib.info import TUNED_PARAMS_PATH
from evaluator.utility.util import get_score, missing_class_corrector


n_trails_simple = 50  # tune 50 times for each simple model
save_param_path = TUNED_PARAMS_PATH + "/evaluators"


def train_lr(params, train_data, test_data, task_type, n_class):
    x_train, y_train = train_data
    if task_type == "regression":
        model = Ridge(**params)
    else:
        multi_class = (
            "multinomial" if task_type == "multiclass_classification" else "auto"
        )
        model = LogisticRegression(**params, n_jobs=-1, multi_class=multi_class)

    const_pred, unique_labels = missing_class_corrector(
        train_data, test_data, task_type
    )
    if const_pred is None:
        model.fit(x_train, y_train)

    score = get_score(model, test_data, task_type, n_class, const_pred, unique_labels)
    return model, score


def tune_lr(
    train_data, val_data, task_type, n_class, dataset=None, store_best_params=True
):
    """
    tune lr with optuna
    when store_best_params is True, we save the best parameters
    otherwise, we return the best model and its score on test data
    note: we only use train and val data for tuning, never use test data
    """

    def lr_objective(trial):
        # configure the model for this trial
        if task_type == "regression":
            alpha = trial.suggest_float("alpha", 0, 10)
            fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
            lr_params = {"alpha": alpha, "fit_intercept": fit_intercept}
        else:
            max_iter = trial.suggest_int("max_iter", 100, 1000)
            C = trial.suggest_float("C", 1e-5, 1e-1, log=True)
            tol = trial.suggest_float("tol", 1e-5, 1e-1, log=True)
            lr_params = {"C": C, "max_iter": max_iter, "tol": tol}

        # train model with train data, test on val data
        model, res = train_lr(lr_params, train_data, val_data, task_type, n_class)
        res = res["rmse"] if task_type == "regression" else res["f1"]
        # save the model
        trial.set_user_attr("best_model", model)
        return res

    study = optuna.create_study(
        direction="maximize" if task_type != "regression" else "minimize",
        sampler=optuna.samplers.TPESampler(seed=0),
    )

    study.optimize(
        lr_objective,
        n_trials=n_trails_simple,
        show_progress_bar=True,
        callbacks=[callback],
    )

    if store_best_params:
        # save the best params on val data
        best_config = study.best_trial.params
        parent_path = save_param_path + "/lr"
        out_path = parent_path + "/{0}.toml".format(dataset)
        os.makedirs(parent_path, exist_ok=True)
        dump_config(best_config, out_path)

    # get the test score of the best model
    best_model = study.user_attrs["best_model"]

    const_pred, unique_labels = missing_class_corrector(train_data, val_data, task_type)
    score = get_score(
        best_model, val_data, task_type, n_class, const_pred, unique_labels
    )
    res = score["rmse"] if task_type == "regression" else score["f1"]
    return best_model, res


def train_svm(params, train_data, test_data, task_type, n_class):
    x_train, y_train = train_data
    if task_type == "regression":
        model = SVR(**params)
    else:
        model = SVC(**params, probability=True)

    const_pred, unique_labels = missing_class_corrector(
        train_data, test_data, task_type
    )
    if const_pred is None:
        model.fit(x_train, y_train)

    score = get_score(model, test_data, task_type, n_class, const_pred, unique_labels)
    return model, score


def tune_svm(
    train_data, val_data, task_type, n_class, dataset=None, store_best_params=True
):
    """
    tune svm with optuna
    when store_best_params is True, we save the best parameters
    otherwise, we return the best model and its score on test data
    note: we only use train and val data for tuning, never use test data
    """

    def svm_objective(trial):
        # configure the model for this trial
        if task_type == "regression":
            C = trial.suggest_float("C", 1e-5, 1e-1, log=True)
            epsilon = trial.suggest_float("epsilon", 1e-5, 1e-1, log=True)
            svm_params = {"C": C, "epsilon": epsilon}
        else:
            C = trial.suggest_float("C", 1e-5, 1e-1, log=True)
            kernel = trial.suggest_categorical(
                "kernel", ["linear", "poly", "rbf", "sigmoid"]
            )
            gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
            svm_params = {"C": C, "kernel": kernel, "gamma": gamma}

        # train model with train data, test on val data
        model, res = train_svm(svm_params, train_data, val_data, task_type, n_class)
        res = res["rmse"] if task_type == "regression" else res["f1"]
        # save the model
        trial.set_user_attr("best_model", model)
        return res

    study = optuna.create_study(
        direction="maximize" if task_type != "regression" else "minimize",
        sampler=optuna.samplers.TPESampler(seed=0),
    )

    study.optimize(
        svm_objective,
        n_trials=n_trails_simple,
        show_progress_bar=True,
        callbacks=[callback],
    )

    if store_best_params:
        # save the best params on val data
        best_config = study.best_trial.params
        parent_path = save_param_path + "/svm"
        out_path = parent_path + "/{0}.toml".format(dataset)
        os.makedirs(parent_path, exist_ok=True)
        dump_config(best_config, out_path)

    # get the test score of the best model
    best_model = study.user_attrs["best_model"]

    const_pred, unique_labels = missing_class_corrector(train_data, val_data, task_type)
    score = get_score(
        best_model, val_data, task_type, n_class, const_pred, unique_labels
    )
    res = score["rmse"] if task_type == "regression" else score["f1"]
    return best_model, res


def train_tree(params, train_data, test_data, task_type, n_class):
    x_train, y_train = train_data

    if task_type == "regression":
        model = DecisionTreeRegressor(**params)
    else:
        model = DecisionTreeClassifier(**params)

    const_pred, unique_labels = missing_class_corrector(
        train_data, test_data, task_type
    )
    if const_pred is None:
        model.fit(x_train, y_train)

    score = get_score(model, test_data, task_type, n_class, const_pred, unique_labels)
    return model, score


def tune_tree(
    train_data, val_data, task_type, n_class, dataset=None, store_best_params=True
):
    """
    tune tree with optuna
    when store_best_params is True, we save the best parameters
    otherwise, we return the best model and its score on test data
    note: we only use train and val data for tuning, never use test data
    """

    def tree_objective(trial):
        # configure the model for this trial
        # same config in both regression and classification
        max_depth = trial.suggest_int("max_depth", 4, 64)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 8)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 8)

        tree_params = {
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
        }

        # train model with train data, test on val data
        model, res = train_tree(tree_params, train_data, val_data, task_type, n_class)
        res = res["rmse"] if task_type == "regression" else res["f1"]
        # save the model
        trial.set_user_attr("best_model", model)
        return res

    study = optuna.create_study(
        direction="maximize" if task_type != "regression" else "minimize",
        sampler=optuna.samplers.TPESampler(seed=0),
    )

    study.optimize(
        tree_objective,
        n_trials=n_trails_simple,
        show_progress_bar=True,
        callbacks=[callback],
    )

    if store_best_params:
        # save the best params
        # used for tune best parameters for real data
        best_config = study.best_trial.params
        parent_path = save_param_path + "/tree"
        out_path = parent_path + "/{0}.toml".format(dataset)
        os.makedirs(parent_path, exist_ok=True)
        dump_config(best_config, out_path)

    # get the test score of the best model
    best_model = study.user_attrs["best_model"]

    const_pred, unique_labels = missing_class_corrector(train_data, val_data, task_type)
    score = get_score(
        best_model, val_data, task_type, n_class, const_pred, unique_labels
    )
    res = score["rmse"] if task_type == "regression" else score["f1"]
    return best_model, res


def train_rf(params, train_data, test_data, task_type, n_class):
    x_train, y_train = train_data
    if task_type == "regression":
        model = RandomForestRegressor(**params)
    else:
        model = RandomForestClassifier(**params)

    const_pred, unique_labels = missing_class_corrector(
        train_data, test_data, task_type
    )
    if const_pred is None:
        model.fit(x_train, y_train)

    score = get_score(model, test_data, task_type, n_class, const_pred, unique_labels)
    return model, score


def tune_rf(
    train_data, val_data, task_type, n_class, dataset=None, store_best_params=True
):
    """
    tune random forest with optuna
    when store_best_params is True, we save the best parameters
    otherwise, we return the best model and its score on test data
    note: we only use train and val data for tuning, never use test data
    """

    def rf_objective(trial):
        # configure the model for this trial
        # same config in both regression and classification
        n_estimators = trial.suggest_int("n_estimators", 10, 200)
        max_depth = trial.suggest_int("max_depth", 4, 64)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 8)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 8)

        rf_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
        }

        # train model with train data, test on val data
        model, res = train_rf(rf_params, train_data, val_data, task_type, n_class)
        res = res["rmse"] if task_type == "regression" else res["f1"]

        # save the model
        trial.set_user_attr("best_model", model)
        return res

    study = optuna.create_study(
        direction="maximize" if task_type != "regression" else "minimize",
        sampler=optuna.samplers.TPESampler(seed=0),
    )

    study.optimize(
        rf_objective,
        n_trials=n_trails_simple,
        show_progress_bar=True,
        callbacks=[callback],
    )
    if store_best_params:
        # save the best params
        # used for tune best parameters for real data
        best_config = study.best_trial.params
        parent_path = save_param_path + "/rf"
        out_path = parent_path + "/{0}.toml".format(dataset)
        os.makedirs(parent_path, exist_ok=True)
        dump_config(best_config, out_path)
    # get the test score of the best model
    best_model = study.user_attrs["best_model"]

    const_pred, unique_labels = missing_class_corrector(train_data, val_data, task_type)
    score = get_score(
        best_model, val_data, task_type, n_class, const_pred, unique_labels
    )
    res = score["rmse"] if task_type == "regression" else score["f1"]
    return best_model, res


def train_mlp(params, train_data, test_data, task_type, n_class):
    x_train, y_train = train_data

    if task_type == "regression":
        model = MLPRegressor(**params)
    else:
        model = MLPClassifier(**params)

    const_pred, unique_labels = missing_class_corrector(
        train_data, test_data, task_type
    )
    if const_pred is None:
        model.fit(x_train, y_train)

    score = get_score(model, test_data, task_type, n_class, const_pred, unique_labels)
    return model, score


def tune_mlp(
    train_data, val_data, task_type, n_class, dataset=None, store_best_params=True
):
    """
    tune mlp with optuna
    when store_best_params is True, we save the best parameters
    otherwise, we return the best model and its score on test data
    note: we only use train and val data for tuning, never use test data
    """

    def mlp_objective(trial):
        # configure the model for this trial
        # same config in both regression and classification
        max_iter = trial.suggest_int("max_iter", 50, 200)
        alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)

        mlp_params = {
            "max_iter": max_iter,
            "alpha": alpha,
        }

        model, res = train_mlp(mlp_params, train_data, val_data, task_type, n_class)
        res = res["rmse"] if task_type == "regression" else res["f1"]

        # save the model
        trial.set_user_attr("best_model", model)
        return res

    study = optuna.create_study(
        direction="maximize" if task_type != "regression" else "minimize",
        sampler=optuna.samplers.TPESampler(seed=0),
    )

    study.optimize(
        mlp_objective,
        n_trials=n_trails_simple,
        show_progress_bar=True,
        callbacks=[callback],
    )

    if store_best_params:
        # save the best params
        # used for tune best parameters for real data
        best_config = study.best_trial.params
        parent_path = save_param_path + "/mlp"
        out_path = parent_path + "/{0}.toml".format(dataset)
        os.makedirs(parent_path, exist_ok=True)
        dump_config(best_config, out_path)

    # get the test score of the best model
    best_model = study.user_attrs["best_model"]

    const_pred, unique_labels = missing_class_corrector(train_data, val_data, task_type)
    score = get_score(
        best_model, val_data, task_type, n_class, const_pred, unique_labels
    )
    res = score["rmse"] if task_type == "regression" else score["f1"]
    return best_model, res
