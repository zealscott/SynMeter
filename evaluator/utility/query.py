import numpy as np
from itertools import product
from lib.commons import improve_reproducibility
import time
import contextlib

def sample_random_range(min_val, max_val, seed):
    """
    sample a range from min_val to max_val
    return a list (min, max)
    """
    min_val = np.random.uniform(min_val, max_val)
    max_val = np.random.uniform(min_val, max_val)
    # let min_val < max_val
    if min_val > max_val:
        min_val, max_val = max_val, min_val
    return [min_val, max_val]


def cal_cond_prob(data, cond_cols):
    """
    Calculate conditional probability for specific attribute
    data: pd.DataFrame
    cond_cols: dict {col_name: col_value}
    """
    if len(cond_cols) == 0:
        return 1.0
    else:
        _sum = len(data)
        for col_name, col_value in cond_cols.items():
            data = data[data[col_name] == col_value]
            if len(data) == 0:
                return 0.0
        return len(data) / _sum


def cal_query(data, cond_cols):
    """
    Given mixed queries, calculate the probability of the query
    """
    if len(cond_cols) == 0:
        return 1.0
    else:
        _sum = len(data)
        for col_name, col_value in cond_cols.items():
            if isinstance(col_value, list):
                data = data[data[col_name] >= col_value[0]]
                data = data[data[col_name] <= col_value[1]]
            else:
                data = data[data[col_name] == col_value]
            if len(data) == 0:
                return 0.0
        return len(data) / _sum


def construct_cond(cols, values):
    cond_dit = {}
    for col, value in zip(cols, values):
        cond_dit[col] = value
    return cond_dit


def cal_all_cond_prob(data, cols, values):
    """
    Calculate conditional probability for given attributes and all their possible values
    cols: list of column names
    values: list of list of possible values
    """
    probs = []
    if len(cols) == 1:
        cols, values = cols[0], values[0]
        for value in values:
            cond_ditc = {cols: value}
            cond_prob = cal_cond_prob(data, cond_ditc)
            probs.append(cond_prob)
    else:
        for each in product(*values):
            cond_ditc = construct_cond(cols, each)
            cond_prob = cal_cond_prob(data, cond_ditc)
            probs.append(cond_prob)
    return probs


@contextlib.contextmanager
def temp_seed(seed):
    """
    set the temporal seed for numpy random generator
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def sample_queries(discrete_col_value_dict, continuous_col_range_dict, n_way_range, n_samples, seed):
    """
    sample queries from both discrete and continuous columns
    """
    improve_reproducibility(seed)
    discrete_cols = list(discrete_col_value_dict.keys())
    continuous_cols = list(continuous_col_range_dict.keys())
    sampled_cond_dict = []
    while len(sampled_cond_dict) < n_samples:
        cond_dict = {}
        while len(cond_dict) < n_way_range:
            # sample query independently
            new_seed = int(1000 * time.time()) % 2**32
            with temp_seed(new_seed):
                # break the seed locally to avoid the same column being sampled
                col = np.random.choice(discrete_cols + continuous_cols)
                if col in discrete_cols:
                    val = np.random.choice(discrete_col_value_dict[col])
                else:
                    min_val, max_val = continuous_col_range_dict[col]
                    val = sample_random_range(min_val, max_val, seed)
                cond_dict[col] = val
        sampled_cond_dict.append(cond_dict)

    return sampled_cond_dict


def range_query(real_data, syn_data, discrete_col_value_dict, continuous_col_range_dict, n_way_range, n_samples, seed):
    """
    evaluate the range query error
    """
    sampled_cond_dict = sample_queries(discrete_col_value_dict, continuous_col_range_dict, n_way_range, n_samples, seed)
    ans = []
    for cond_dict in sampled_cond_dict:
        real_prob = cal_query(real_data, cond_dict)
        syn_prob = cal_query(syn_data, cond_dict)
        ans.append(np.abs(real_prob - syn_prob))
        # if ans[-1] < 1e-5:
        #     print(cond_dict)
        #     print(real_prob, syn_prob)
    avg_query_error = np.mean(ans)
    return avg_query_error
