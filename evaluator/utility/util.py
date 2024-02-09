import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
from torch import Tensor
from lib.commons import cal_metrics
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

ModuleType = Union[str, Callable[..., nn.Module]]


def split_data_stratify(data, test_size=0.2):
    """
    data: pandas dataframe
    split data into train and test
    use stratify to make sure the distribution of the data/label is the same
    do not use any preprocessing, return dataframe
    """
    # split data into data and label
    label = data["label"]
    data = data.drop(columns=["label"])
    # split data into train, test
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            data, label, test_size=test_size, random_state=0, stratify=label
        )
    except:
        # when the data is too small, we cannot split with stratify
        x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=test_size, random_state=0)
    # combine data and label
    train = pd.concat([x_train, y_train], axis=1)
    test = pd.concat([x_test, y_test], axis=1)

    return train, test


def missing_class_corrector(train_data, test_data, task_type):
    """
    tackle the problem when train with one class or test with unseen class\\
    return constant prediction if the model cannot handle the case
    """
    x_train, y_train = train_data
    x_test, y_test = test_data
    const_pred = None

    if task_type == "regression":
        return None, None

    unique_labels = np.unique(y_train)
    if task_type == "binary_classification" and len(unique_labels) == 1:
        # when train with only one class, the model cannot predict
        # we use a constant prediction
        pred = [unique_labels[0]] * len(x_test)
        pred_prob = np.array([1.0] * len(x_test))
        const_pred = [pred, pred_prob]
    elif task_type == "multiclass_classification":
        if len(unique_labels) < 3 or len(unique_labels) != len(np.unique(y_test)):
            # when train with only one/two class for multiclass_classification, the model cannot predict
            # when multi-class classification and has unseen class in test data
            # which cannot handle by most multi-class classification models
            # we use a constant prediction (same with STaSy)
            pred = [unique_labels[0]] * len(x_test)
            pred_prob = np.array([1.0] * len(x_test))
            const_pred = [pred, pred_prob]

    return const_pred, unique_labels


def get_score(model, test_data, task_type, n_class, const_pred, unique_labels):
    """
    evaluate the model on the test data
    return the score of the model on the test data
    """
    x_arr, y_arr = test_data
    if task_type == "regression":
        y_pred = model.predict(x_arr)
        score = cal_metrics(y_arr, y_pred, task_type)
    else:
        # missing class only happen in classification task
        if const_pred is None:
            # no missing class
            pred = model.predict(x_arr)
            pred_prob = model.predict_proba(x_arr)
            if task_type == "binary_classification" and len(pred_prob.shape) == 3:
                # tackle tab-transformer's output
                pred = pred.squeeze(-1)
                pred_prob = pred_prob.squeeze(axis=2)
                y_arr = y_arr.squeeze()
        else:
            pred, pred_prob = const_pred
        score = cal_metrics(y_arr, pred, task_type, pred_prob, n_class, unique_labels)

    return score


# --------------------------------------------------------------- #
# ------------------- Tab-MLP strong evaluator ------------------ #
# --------------------------------------------------------------- #


def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


class ReGLU(nn.Module):
    """The ReGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = ReGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)


class GEGLU(nn.Module):
    """The GEGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = GEGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)


def _make_nn_module(module_type: ModuleType, *args) -> nn.Module:
    return (
        (
            ReGLU()
            if module_type == "ReGLU"
            else GEGLU()
            if module_type == "GEGLU"
            else getattr(nn, module_type)(*args)
        )
        if isinstance(module_type, str)
        else module_type(*args)
    )


class TabTransformer(nn.Module):
    """The TabTransformer model used in [gorishniy2021revisiting].

    The following scheme describes the architecture:

    .. code-block:: text

          TabTransformer: (in) -> Block -> ... -> Block -> Linear -> (out)
        Block: (in) -> Linear -> Activation -> Dropout -> (out)

    Examples:
        .. testcode::

            x = torch.randn(4, 2)
            module = TabTransformer.make_baseline(x.shape[1], [3, 5], 0.1, 1)
            assert module(x).shape == (len(x), 1)

    References:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
    """

    class Block(nn.Module):
        """The main building block of `TabTransformer`."""

        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            activation: ModuleType,
            dropout: float,
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias)
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            return self.dropout(self.activation(self.linear(x)))

    def __init__(
        self,
        *,
        d_in: int,
        d_layers: List[int],
        dropouts: Union[float, List[float]],
        activation: Union[str, Callable[[], nn.Module]],
        d_out: int,
    ) -> None:
        """
        Note:
            `make_baseline` is the recommended constructor.
        """
        super().__init__()
        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(d_layers)
        assert len(d_layers) == len(dropouts)
        assert activation not in ["ReGLU", "GEGLU"]

        self.blocks = nn.ModuleList(
            [
                TabTransformer.Block(
                    d_in=d_layers[i - 1] if i else d_in,
                    d_out=d,
                    bias=True,
                    activation=activation,
                    dropout=dropout,
                )
                for i, (d, dropout) in enumerate(zip(d_layers, dropouts))
            ]
        )
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    @classmethod
    def make_baseline(
        cls: Type["TabTransformer"],
        d_in: int,
        d_layers: List[int],
        dropout: float,
        d_out: int,
    ) -> "TabTransformer":
        """Create a "baseline" `TabTransformer`.

        This variation of TabTransformer was used in [gorishniy2021revisiting]. Features:

        * :code:`Activation` = :code:`ReLU`
        * all linear layers except for the first one and the last one are of the same dimension
        * the dropout rate is the same for all dropout layers

        Args:
            d_in: the input size
            d_layers: the dimensions of the linear layers. If there are more than two
                layers, then all of them except for the first and the last ones must
                have the same dimension. Valid examples: :code:`[]`, :code:`[8]`,
                :code:`[8, 16]`, :code:`[2, 2, 2, 2]`, :code:`[1, 2, 2, 4]`. Invalid
                example: :code:`[1, 2, 3, 4]`.
            dropout: the dropout rate for all hidden layers
            d_out: the output size
        Returns:
            TabTransformer

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        assert isinstance(dropout, float)
        if len(d_layers) > 2:
            assert len(set(d_layers[1:-1])) == 1, (
                "if d_layers contains more than two elements, then"
                " all elements except for the first and the last ones must be equal."
            )
        return TabTransformer(
            d_in=d_in,
            d_layers=d_layers,  # type: ignore
            dropouts=dropout,
            activation="ReLU",
            d_out=d_out,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        # squeeze the last dimension if d_out == 1
        # if x.shape[-1] == 1:
        #     x = x.squeeze(-1)
        return x


# --------------------------------------------------------------- #
# -------------------------- Optuna helper ---------------------- #
# --------------------------------------------------------------- #


# https://stackoverflow.com/questions/62144904/python-how-to-retrive-the-best-model-from-optuna-lightgbm-study
def callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_model", value=trial.user_attrs["best_model"])
