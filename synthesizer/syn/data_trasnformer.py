"""DataTransformer module."""
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, LabelEncoder, OrdinalEncoder


class DataTransformer(object):
    def __init__(self,max_bins = 100):
        self.max_bins = max_bins

    def fit(self, data, discrete_columns):
        """
        data: pandas.DataFrame
        discrete_columns: list
        transform discrete columns to 0 to n-1 encoding
        transform continuous columns with binning (max 100 bins)
        """
        self.original_columns = data.columns
        self.discrete_columns = discrete_columns
        self.continuous_columns = [col for col in data.columns if col not in discrete_columns]
        self.domain = {}

        # transofrm discrete columns to ordinal encoding column by column
        # the one-hot encoding is handeled in the model
        self.cat_transformers = {}
        for col in self.discrete_columns:
            col_data = data[col].values.reshape(-1, 1)
            self.cat_transformers[col] = LabelEncoder()
            self.cat_transformers[col].fit(col_data.ravel())
            self.domain[col] = len(self.cat_transformers[col].classes_)

        # transform continuous columns to binning column by column
        self.num_transformers = {}
        for col in self.continuous_columns:
            col_data = data[col].values.reshape(-1, 1)
            # count the number of unique values
            n_values = len(np.unique(col_data))
            # get the distribution of the values
            value_counts = np.unique(col_data, return_counts=True)[1]
            n_bins = min(n_values, self.max_bins)
            self.num_transformers[col] = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform",subsample = 500000)
            self.num_transformers[col].fit(col_data)
            self.domain[col] = self.num_transformers[col].n_bins_[0]
                
    def transform(self, data):
        """
        data: pandas.DataFrame
        transform data to a new dataframe with ordinal integer encoding
        return a domain of the transformed data
        """
        trans_data = []
        
        for col in data.columns:
            col_data = data[col].values.reshape(-1, 1)
            if col in self.discrete_columns:
                trans_data.append(self.cat_transformers[col].transform(col_data.ravel()).reshape(-1, 1))
            else:
                trans_data.append(self.num_transformers[col].transform(col_data).reshape(-1, 1))
        
        # convert to pandas dataframe
        trans_data = np.concatenate(trans_data, axis=1).astype(int)
        trans_data_pd = pd.DataFrame(trans_data, columns=data.columns)

        return trans_data_pd

    def inverse_transform(self, data):
        """
        data: pandas.DataFrame
        inverse transform data to original data
        """
        data_inv = []
        for col in data.columns:
            col_data = data[col].values.reshape(-1, 1)
            if col in self.discrete_columns:
                data_inv.append(self.cat_transformers[col].inverse_transform(col_data).reshape(-1, 1))
            else:
                data_inv.append(self.num_transformers[col].inverse_transform(col_data).reshape(-1, 1))
        
        # convert to pandas dataframe
        data_inv = np.concatenate(data_inv, axis=1)
        data_inv_pd = pd.DataFrame(data_inv, columns=data.columns)
        # rerange the columns
        data_inv_pd = data_inv_pd[self.original_columns]
        return data_inv_pd

    def fit_transform(self, data, discrete_columns):
        """
        data: pandas.DataFrame
        discrete_columns: list
        fit and transform data
        return transformed data and domain
        """
        self.fit(data, discrete_columns)
        return self.transform(data)

    def get_mapping(self):
        """
        get the encoding domain
        """
        encode_mapping = {}
        for col in self.original_columns:
            if col in self.discrete_columns:
                # get the mapping of the encoder
                encode_mapping[col] = dict(zip(self.cat_transformers[col].classes_, range(len(self.cat_transformers[col].classes_))))
            else:
                # get the bin range mapping of the encoder 
                bin_edges = self.num_transformers[col].bin_edges_[0][1:-1]
                # each range is (left, right)
                bin_intervals= [(-np.inf, bin_edges[0])]
                for i in range(len(bin_edges)-1):
                    bin_intervals.append((bin_edges[i], bin_edges[i+1]))
                bin_intervals.append((bin_edges[-1], np.inf))
                encode_mapping[col] = dict(zip(bin_intervals, range(len(bin_intervals))))
                assert len(encode_mapping[col]) == self.domain[col]
                
        
        return encode_mapping
                