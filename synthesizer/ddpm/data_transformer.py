"""DataTransformer module."""
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer, StandardScaler, LabelEncoder, OrdinalEncoder
import copy

class DataTransformer(object):
    def __init__(self, y_cond):
        self.cond_y = y_cond

    def fit(self, data, discrete_columns):
        """
        data: pandas.DataFrame
        discrete_columns: list
        transform discrete columns to one-hot encoding
        transform continuous columns with quantile transform
        transform label column to one-hot encoding/stardard encoding
        """
        self.original_columns = copy.deepcopy(data.columns)
        self.discrete_columns = copy.deepcopy(discrete_columns)
        self.continuous_columns = [col for col in data.columns if col not in self.discrete_columns]
        if self.cond_y:
            # only used in classification task
            assert "label" in self.discrete_columns
            self.discrete_columns.remove("label")

        # transofrm discrete columns to ordinal encoding column by column
        # the one-hot encoding is handeled in the model
        self.cat_transformers = {}
        for col in self.discrete_columns:
            col_data = data[col].values.reshape(-1, 1)
            # count the number of unique values
            unknown_value = len(np.unique(col_data))
            self.cat_transformers[col] = OrdinalEncoder(
                handle_unknown="use_encoded_value",  # type: ignore[code]
                unknown_value=unknown_value,  # type: ignore[code]
                dtype="int64",  # type: ignore[code]
            )
            # fit the data
            self.cat_transformers[col].fit(col_data)

        # transform continuous columns to quantile column by column
        self.num_transformers = {}
        for col in self.continuous_columns:
            col_data = data[col].values.reshape(-1, 1)
            self.num_transformers[col] = QuantileTransformer()
            # fit the data
            self.num_transformers[col].fit(col_data)

        self.label_transformer = None
        if self.cond_y:
            # classification task
            col_data = data["label"].values.reshape(-1, 1).ravel()
            self.label_transformer = LabelEncoder()
            self.label_transformer.fit(col_data)
        else:
            # regression task, standardize the label
            col_data = data["label"].values.reshape(-1, 1)
            self.label_transformer = StandardScaler()
            self.label_transformer.fit(col_data.reshape(-1, 1))

    def transform(self, data):
        """
        data: pandas.DataFrame
        transform data to one-hot encoding and quantile transformed data
        """
        if self.continuous_columns:
            # transform continuous columns to quantile transformed data
            num_data = np.concatenate(
                [
                    self.num_transformers[col].transform(data[col].values.reshape(-1, 1))
                    for col in self.continuous_columns
                ],
                axis=1,
            )

        # transform discrete columns to one-hot encoding
        if self.discrete_columns:
            cat_data = np.concatenate(
                [
                    self.cat_transformers[col].transform(data[col].values.reshape(-1, 1))
                    for col in self.discrete_columns
                ],
                axis=1,
            )

        # transform label column to one-hot encoding/stardard encoding
        label_col = data["label"].values.reshape(-1, 1)
        if self.cond_y:
            # classification task
            transformed_label = self.label_transformer.transform(label_col.ravel())
        else:
            transformed_label = self.label_transformer.transform(label_col)
        _, empirical_class_dist = torch.unique(torch.from_numpy(transformed_label), return_counts=True)
        self.empirical_class_dist = empirical_class_dist.float()

        self.cal_dimension()

        # concatenate transformed data into numpy
        if self.continuous_columns and self.discrete_columns:
            transformed_data = np.concatenate([num_data, cat_data], axis=1)
        elif self.continuous_columns:
            transformed_data = num_data
        elif self.discrete_columns:
            transformed_data = cat_data

        return transformed_data, transformed_label

    def inverse_transform(self, data, label):
        """
        data: numpy.ndarray
        label: only used in classification task
        inverse transform data to original data
        """
        # inverse transform data according to the order of transform
        num_data = data[:, : self.num_dim].numpy()
        cat_data = data[:, self.num_dim :].numpy()

        # inverse transform continuous columns
        num_data_inv = []
        for col in self.continuous_columns:
            num_data_inv.append(
                self.num_transformers[col].inverse_transform(
                    num_data[:, self.continuous_columns.index(col)].reshape(-1, 1)
                )
            )

        # inverse transform discrete columns, each column is the ordinal encoding
        cat_data_inv = []
        for col in self.discrete_columns:
            cat_data_inv.append(
                self.cat_transformers[col].inverse_transform(
                    cat_data[:, self.discrete_columns.index(col)].reshape(-1, 1)
                )
            )

        label_data_inv = self.label_transformer.inverse_transform(label.reshape(-1, 1))

        # concatenate inverse transformed data into numpy
        if self.cond_y == False:
            data_inv = np.concatenate(num_data_inv + cat_data_inv, axis=1)
            data_inv_pd = pd.DataFrame(data_inv, columns=self.continuous_columns + self.discrete_columns)
        else:
            data_inv = np.concatenate(num_data_inv + cat_data_inv + [label_data_inv.reshape(-1, 1)], axis=1)
            data_inv_pd = pd.DataFrame(
                data_inv, columns=self.continuous_columns + self.discrete_columns + ["label"]
            )

        # rearrange the order of columns
        data_inv_pd = data_inv_pd[self.original_columns]

        return data_inv_pd

    def fit_transform(self, data, discrete_columns):
        """
        data: pandas.DataFrame
        discrete_columns: list
        fit and transform data
        """
        self.fit(data, discrete_columns)
        return self.transform(data)

    def cal_dimension(self):
        """
        calculate the dimension of transformed data of each column
        """
        self.cat_dim = {}
        for col in self.discrete_columns:
            self.cat_dim[col] = self.cat_transformers[col].categories_[0].shape[0]
        self.num_dim = len(self.continuous_columns)

    def get_dim(self):
        """
        return the dimension of transformed data
        """
        return self.num_dim + sum(self.cat_dim.values())

    def get_cat_dim(self):
        """
        return the dimension of each transformed discrete columns [k1,k2,...,kn]
        """
        if self.discrete_columns:
            return np.array([self.cat_dim[col] for col in self.discrete_columns])
        else:
            return np.array([0])

    def get_num_dim(self):
        """
        return the dimension of transformed continuous columns
        """
        return self.num_dim

    def get_label_dim(self):
        """
        return the dimension of label column
        """
        if self.cond_y:
            return len(self.label_transformer.classes_)
        else:
            return 1
