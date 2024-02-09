import numpy as np
from . import FactoredInference, Dataset, Domain
from scipy import sparse
from disjoint_set import DisjointSet
import networkx as nx
import itertools
from .cdp2adp import cdp_rho
from scipy.special import logsumexp
from lib.commons import load_config
from lib.commons import read_csv
from .data_trasnformer import DataTransformer
import pandas as pd

"""
This is a generalization of the winning mechanism from the 
2018 NIST Differential Privacy Synthetic Data Competition.

Unlike the original implementation, this one can work for any discrete dataset,
and does not rely on public provisional data for measurement selection.  
"""


def MST(
    data,
    epsilon,
    delta,
):
    rho = cdp_rho(epsilon, delta)
    sigma = np.sqrt(3 / (2 * rho))
    cliques = [(col,) for col in data.domain]
    log1 = measure(data, cliques, sigma)
    data, log1, supports = compress_domain(data, log1)
    cliques = list(itertools.combinations(data.domain, 2))[:30]
    log2 = measure(data, cliques, sigma)
    engine = FactoredInference(data.domain, iters=5000)
    est = engine.estimate(log1 + log2)
    return est, supports


def MST_private(data, epsilon, delta, nums, nun_iters, cliques2, device):
    """
    use 3-way marginals to learn a PGM
    2/3 cliques are given measurements
    """
    rho = cdp_rho(epsilon, delta)
    sigma = np.sqrt(3 / (2 * rho))
    cliques = [(col,) for col in data.domain]
    log1 = measure(data, cliques, sigma)
    data, log1, supports = compress_domain(data, log1)
    cliques = list(itertools.combinations(data.domain, 2))
    # measure the pairwise marginals
    log2 = measure(data, cliques2, sigma)
    log = log1 + log2
    engine = FactoredInference(data.domain, iters=nun_iters)
    est = engine.estimate(log)
    return est, supports


def MST_no_privacy(data, epsilon, delta, nums, nun_iters, cliques2, cliques3, device):
    """
    use 3-way marginals to learn a PGM
    2/3 cliques are given measurements
    """
    bi_num, tri_num = nums
    rho = cdp_rho(epsilon, delta)
    sigma = np.sqrt(3 / (2 * rho))
    cliques = [(col,) for col in data.domain]
    log1 = measure(data, cliques, sigma)
    data, log1, supports = compress_domain(data, log1)
    cliques = list(itertools.combinations(data.domain, 2))
    # measure the pairwise marginals
    log2 = measure(data, cliques2, sigma)
    # measure the 3-way marginals
    if len(cliques3) > 0:
        log3 = measure(data, cliques3, sigma)
        log = log1 + log2 + log3
    else:
        log = log1 + log2
    engine = FactoredInference(data.domain, iters=nun_iters)
    est = engine.estimate(log)
    return est, supports

def measure(data, cliques, sigma, weights=None):
    if weights is None:
        weights = np.ones(len(cliques))
    weights = np.array(weights) / np.linalg.norm(weights)
    measurements = []
    for proj, wgt in zip(cliques, weights):
        x = data.project(proj).datavector()
        y = x + np.random.normal(loc=0, scale=sigma / wgt, size=x.size)
        Q = sparse.eye(x.size)
        measurements.append((Q, y, sigma / wgt, proj))
    return measurements


def compress_domain(data, measurements):
    supports = {}
    new_measurements = []
    for Q, y, sigma, proj in measurements:
        col = proj[0]
        sup = y >= 3 * sigma
        supports[col] = sup
        if supports[col].sum() == y.size:
            new_measurements.append((Q, y, sigma, proj))
        else:  # need to re-express measurement over the new domain
            y2 = np.append(y[sup], y[~sup].sum())
            I2 = np.ones(y2.size)
            I2[-1] = 1.0 / np.sqrt(y.size - y2.size + 1.0)
            y2[-1] /= np.sqrt(y.size - y2.size + 1.0)
            I2 = sparse.diags(I2)
            new_measurements.append((I2, y2, sigma, proj))
    # undo_compress_fn = lambda data: reverse_data(data, supports)
    # return transform_data(data, supports), new_measurements, undo_compress_fn
    return transform_data(data, supports), new_measurements, supports


def exponential_mechanism(q, eps, sensitivity, prng=np.random, monotonic=False):
    coef = 1.0 if monotonic else 0.5
    scores = coef * eps / sensitivity * q
    probas = np.exp(scores - logsumexp(scores))
    return prng.choice(q.size, p=probas)


def select(data, rho, measurement_log, cliques=[]):
    engine = FactoredInference(data.domain, iters=1000)
    est = engine.estimate(measurement_log)

    weights = {}
    candidates = list(itertools.combinations(data.domain.attrs, 2))
    for a, b in candidates:
        xhat = est.project([a, b]).datavector()
        x = data.project([a, b]).datavector()
        weights[a, b] = np.linalg.norm(x - xhat, 1)

    T = nx.Graph()
    T.add_nodes_from(data.domain.attrs)
    ds = DisjointSet()

    for e in cliques:
        T.add_edge(*e)
        ds.union(*e)

    r = len(list(nx.connected_components(T)))
    epsilon = np.sqrt(8 * rho / (r - 1))
    for i in range(r - 1):
        candidates = [e for e in candidates if not ds.connected(*e)]
        wgts = np.array([weights[e] for e in candidates])
        idx = exponential_mechanism(wgts, epsilon, sensitivity=1.0)
        e = candidates[idx]
        T.add_edge(*e)
        ds.union(*e)

    return list(T.edges)


def transform_data(data, supports):
    df = data.df.copy()
    newdom = {}
    for col in data.domain:
        support = supports[col]
        size = support.sum()
        newdom[col] = int(size)
        if size < support.size:
            newdom[col] += 1
        mapping = {}
        idx = 0
        for i in range(support.size):
            mapping[i] = size
            if support[i]:
                mapping[i] = idx
                idx += 1
        assert idx == size
        df[col] = df[col].map(mapping)
    newdom = Domain.fromdict(newdom)
    return Dataset(df, newdom)


def reverse_data(data, supports):
    df = data.df.copy()
    newdom = {}
    for col in data.domain:
        support = supports[col]
        mx = support.sum()
        newdom[col] = int(support.size)
        idx, extra = np.where(support)[0], np.where(~support)[0]
        mask = df[col] == mx
        if extra.size == 0:
            pass
        else:
            df.loc[mask, col] = np.random.choice(extra, mask.sum())
        df.loc[~mask, col] = idx[df.loc[~mask, col]]
    newdom = Domain.fromdict(newdom)
    return Dataset(df, newdom)


def train_wrapper_PGM(args, cuda, tune = False):
    path_params = args["path_params"]
    model_params = args["model_params"]

    epsilon = model_params["epsilon"]
    delta = model_params["delta"]
    max_bins = model_params["max_bins"]
    bi_nums = model_params["bi_nums"]
    tri_nums = model_params["tri_nums"]
    num_iters = model_params["num_iters"]
    cliques2 = model_params["2_cliques"]
    cliques3 = model_params["3_cliques"]

    device = "cuda:" + cuda

    # prepare data
    train_data_pd, meta_data, discrete_columns = read_csv(path_params["train_data"], path_params["meta_data"])
    val_data_pd, _, _ = read_csv(path_params["val_data"], path_params["meta_data"])
    if tune:
        data_pd = train_data_pd
    else:
        # combine train and val data
        data_pd = pd.concat([train_data_pd, val_data_pd], ignore_index=True, sort=False)

    data_transformer = DataTransformer(max_bins)

    transformed_data, domain = data_transformer.fit_transform(data_pd, discrete_columns)

    data = Dataset.load(transformed_data, domain)

    learned_pgm, supports = MST_no_privacy(
        data, epsilon, delta, [bi_nums, tri_nums], num_iters, cliques2, cliques3, device
    )

    model = {}
    model["learned_pgm"] = learned_pgm
    model["data_transformer"] = data_transformer
    model["supports"] = supports

    return model



def train_wrapper_PGM_private(args, cuda, tune = False):
    path_params = args["path_params"]
    model_params = args["model_params"]

    epsilon = model_params["epsilon"]
    delta = model_params["delta"]
    max_bins = model_params["max_bins"]
    bi_nums = model_params["bi_nums"]
    num_iters = model_params["num_iters"]
    cliques2 = model_params["2_cliques"]

    device = "cuda:" + cuda

    # prepare data
    train_data_pd, meta_data, discrete_columns = read_csv(path_params["train_data"], path_params["meta_data"])
    val_data_pd, _, _ = read_csv(path_params["val_data"], path_params["meta_data"])
    if tune:
        data_pd = train_data_pd
    else:
        # combine train and val data
        data_pd = pd.concat([train_data_pd, val_data_pd], ignore_index=True, sort=False)

    data_transformer = DataTransformer(max_bins)

    transformed_data, domain = data_transformer.fit_transform(data_pd, discrete_columns)

    data = Dataset.load(transformed_data, domain)

    learned_pgm, supports = MST_private(
        data, epsilon, delta, bi_nums, num_iters, cliques2, device
    )

    model = {}
    model["learned_pgm"] = learned_pgm
    model["data_transformer"] = data_transformer
    model["supports"] = supports

    return model