import numpy as np
from lib.commons import dump_config, cal_metrics, rmse, f1
from evaluator.utility.util import TabTransformer, callback
import optuna
import os
from skorch.regressor import NeuralNetRegressor
from skorch.classifier import NeuralNetClassifier
from skorch.dataset import Dataset as SkDataset
from skorch.callbacks import EarlyStopping, EpochScoring
from skorch.helper import predefined_split
from torch.nn import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import AdamW
from lib.info import TUNED_PARAMS_PATH
from evaluator.utility.util import get_score, missing_class_corrector


def train_tab_transformer(params, train_data, test_data, task_type, n_class, device):
    if task_type == "regression" or task_type == "binary_classification":
        train_data[1] = train_data[1].reshape(-1, 1).astype(np.float32)
        test_data[1] = test_data[1].reshape(-1, 1).astype(np.float32)
        
    es = EarlyStopping(monitor="valid_loss", patience=16)
    val_ds = SkDataset(test_data[0], test_data[1])

    # set up model
    model = TabTransformer.make_baseline(
        d_in=params["d_in"], d_out=params["d_out"], d_layers=params["d_layers"], dropout=params["dropout"]
    )

    if task_type == "regression":
        net = NeuralNetRegressor(
            model,
            criterion=MSELoss,
            optimizer=AdamW,
            lr=params["lr"],
            optimizer__weight_decay=params["weight_decay"],
            batch_size=params["batch_size"],
            max_epochs=1000,
            train_split=predefined_split(val_ds),
            iterator_train__shuffle=True,
            device=device,
            callbacks=[es, EpochScoring(rmse, lower_is_better=True)],
        )
    else:
        net = NeuralNetClassifier(
            model,
            criterion=BCEWithLogitsLoss if task_type == "binary_classification" else CrossEntropyLoss,
            optimizer=AdamW,
            lr=params["lr"],
            optimizer__weight_decay=params["weight_decay"],
            batch_size=params["batch_size"],
            max_epochs=1000,
            train_split=predefined_split(val_ds),
            iterator_train__shuffle=True,
            device=device,
            callbacks=[es, EpochScoring(f1, lower_is_better=False)],
        )

    const_pred, unique_labels = missing_class_corrector(train_data, test_data, task_type)
    if const_pred is None:
        net.fit(X=train_data[0], y=train_data[1])

    score = get_score(net, test_data, task_type, n_class, const_pred, unique_labels)
    return net, score


def _suggest_mlp_layers(trial):
    def suggest_dim(name):
        t = trial.suggest_int(name, d_min, d_max)
        return 2**t

    min_n_layers, max_n_layers, d_min, d_max = 1, 4, 7, 10
    n_layers = 2 * trial.suggest_int("n_layers", min_n_layers, max_n_layers)
    d_first = [suggest_dim("d_first")] if n_layers else []
    d_middle = [suggest_dim("d_middle")] * (n_layers - 2) if n_layers > 2 else []
    d_last = [suggest_dim("d_last")] if n_layers > 1 else []
    d_layers = d_first + d_middle + d_last
    return d_layers


def tune_tab_transformer(
    train_data, val_data, task_type, n_class, dataset=None, store_best_params=True, device=None
):
    """
    tune tab transoformer with optuna
    when store_best_params is True, we save the best parameters
    otherwise, we return the best model and its score on test data
    note: we only use train and val data for tuning, never use test data
    """

    def tab_transformer_objective(trial):
        # configure the model for this trial
        d_layers = _suggest_mlp_layers(trial)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        batch_size = trial.suggest_int("batch_size", 256, 4096)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        n_classes = len(np.unique(y_train)) if task_type == "multiclass_classification" else 1

        # get MLP configures
        mlp_params = {}
        mlp_params["d_layers"] = d_layers
        mlp_params["dropout"] = dropout
        mlp_params["d_in"] = x_train.shape[1]
        mlp_params["d_out"] = n_classes
        # store training configures
        mlp_params["lr"] = lr
        mlp_params["batch_size"] = batch_size
        mlp_params["weight_decay"] = weight_decay
        # store configures
        trial.set_user_attr("config", mlp_params)

        # train model with train data, test on val data
        net, res = train_tab_transformer(mlp_params, train_data, val_data, task_type, n_class, device)
        res = res["rmse"] if task_type == "regression" else res["f1"]
        # save the model
        trial.set_user_attr("best_model", net)

        return res

    study = optuna.create_study(
        direction="maximize" if task_type != "regression" else "minimize",
        sampler=optuna.samplers.TPESampler(seed=0),
    )
    
    # get data for tune
    x_train, y_train = train_data
    x_valid, y_valid = val_data
    if task_type == "regression" or task_type == "binary_classification":
        train_data[1] = y_train.reshape(-1, 1).astype(np.float32)
        val_data[1] = y_valid.reshape(-1, 1).astype(np.float32)

    study.optimize(tab_transformer_objective, n_trials=100, show_progress_bar=True, callbacks=[callback])

    if store_best_params:
        # save the best params
        # used for tune best parameters for real data
        best_config = study.best_trial.user_attrs["config"]
        parent_path = TUNED_PARAMS_PATH + "/evaluators/tab_transformer"
        out_path = parent_path + "/{0}.toml".format(dataset)
        os.makedirs(parent_path, exist_ok=True)
        dump_config(best_config, out_path)
    
    # get the test score of the best model
    best_model = study.user_attrs["best_model"]

    const_pred, unique_labels = missing_class_corrector(train_data, val_data, task_type)
    score = get_score(best_model, val_data, task_type, n_class, const_pred, unique_labels)
    res = score["rmse"] if task_type == "regression" else score["f1"]
    return best_model, res
