import numpy as np
import pandas as pd

def validate_discrete_columns(train_data, discrete_columns):
    """Check whether ``discrete_columns`` exists in ``train_data``.

    Args:
        train_data (numpy.ndarray or pandas.DataFrame):
            Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
        discrete_columns (list-like):
            List of discrete columns to be used to generate the Conditional
            Vector. If ``train_data`` is a Numpy array, this list should
            contain the integer indices of the columns. Otherwise, if it is
            a ``pandas.DataFrame``, this list should contain the column names.
    """
    if isinstance(train_data, pd.DataFrame):
        invalid_columns = set(discrete_columns) - set(train_data.columns)
    elif isinstance(train_data, np.ndarray):
        invalid_columns = []
        for column in discrete_columns:
            if column < 0 or column >= train_data.shape[1]:
                invalid_columns.append(column)
    else:
        raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

    if invalid_columns:
        raise ValueError(f'Invalid columns found: {invalid_columns}')