import numpy as np
import os

from torch.utils.data import Dataset
import torch

class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat):
        self.X_num = X_num
        self.X_cat = X_cat

    def __getitem__(self, index):
        this_num = self.X_num[index] if self.X_num is not None else None
        this_cat = self.X_cat[index] if self.X_cat is not None else None

        sample = (this_num, this_cat)

        return sample

    def __len__(self):
        return self.X_num.shape[0] if self.X_num is not None else self.X_cat.shape[0]


def my_collate(batch):
    """
    handle the case where the batch contains None
    """
    batch_num = [item[0] for item in batch if item[0] is not None]
    batch_cat = [item[1] for item in batch if item[1] is not None]
    # Convert lists to tensors
    batch_num = torch.stack(batch_num) if batch_num else None
    batch_cat = torch.stack(batch_cat) if batch_cat else None

    return batch_num, batch_cat


def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for target, source in zip(target_params, source_params):
        target.detach().mul_(rate).add_(source.detach(), alpha=1 - rate)


def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)
