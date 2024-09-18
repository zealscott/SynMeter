"""DataTransformer module."""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer, StandardScaler, LabelEncoder, OrdinalEncoder
import copy


class DataTransformer(object):
    def __init__(self):
        pass

    def fit(self, data, discrete_columns):
        """
        data: pandas.DataFrame
        discrete_columns: list
        transform discrete columns to orginal encoding
        transform continuous columns with quantile transform
        """
        self.original_columns = copy.deepcopy(data.columns)
        self.discrete_columns = copy.deepcopy(discrete_columns)
        self.continuous_columns = [col for col in data.columns if col not in self.discrete_columns]

        # transofrm discrete columns to ordinal encoding column by column
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
            self.num_transformers[col] = QuantileTransformer(
                output_distribution="normal", n_quantiles=max(min(len(data) // 30, 1000), 10), subsample=int(1e9)
            )

            # fit the data
            self.num_transformers[col].fit(col_data)

    def transform(self, data):
        """
        data: pandas.DataFrame
        transform data to ordinal encoding and quantile transformed data
        return: num_data, cat_data
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

        # transform discrete columns to ordinal encoding
        if self.discrete_columns:
            cat_data = np.concatenate(
                [
                    self.cat_transformers[col].transform(data[col].values.reshape(-1, 1))
                    for col in self.discrete_columns
                ],
                axis=1,
            )

        self.cal_dimension()

        # concatenate transformed data into numpy
        if self.continuous_columns and self.discrete_columns:
            return num_data, cat_data
        elif self.continuous_columns:
            return num_data, None
        elif self.discrete_columns:
            return None, cat_data
        else:
            raise ValueError("No columns in the data")

    def inverse_transform(self, data):
        """
        data: numpy.ndarray
        inverse transform data to original data
        """
        # inverse transform data according to the order of transform
        num_data = data[:, : self.num_dim]
        cat_data = data[:, self.num_dim :]

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

        data_inv = np.concatenate(num_data_inv + cat_data_inv, axis=1)
        data_inv_pd = pd.DataFrame(data_inv, columns=self.continuous_columns + self.discrete_columns)
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
            return [self.cat_dim[col] for col in self.discrete_columns]
        else:
            return None

    def get_num_dim(self):
        """
        return the dimension of transformed continuous columns
        """
        return self.num_dim

    def save(self, path):
        """
        save the transformer to the path with pickle
        """
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f)
