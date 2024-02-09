import torch
from torch import nn


class Residual(nn.Module):
    """Residual layer"""

    def __init__(self, i, o):
        super().__init__()
        self.fc = nn.Linear(i, o)
        # self.bn = nn.BatchNorm1d(o)
        self.bn = nn.GroupNorm(1, o)  # Use privacy-safe groupnorm over batchnorm
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, x], dim=1)


class Generator(nn.Module):
    """Based on the CTGAN implementation at
    https://github.com/sdv-dev/CTGAN/blob/master/ctgan/synthesizers/ctgan.py
    """

    def __init__(self, embedding_dim, data_dim, gen_dims=(256, 256)):
        super().__init__()
        dim = embedding_dim
        seq = []
        for item in list(gen_dims):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(nn.Linear(dim, data_dim))
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)


class MixedTypeGenerator(Generator):
    def __init__(
        self,
        embedding_dim,
        data_dim,
        gen_dims=(256, 256),
        predict_noise=True,
        categorical_start_idx=None,
        cat_counts=None,
    ):
        # Initialise parent (Generator) with the parameters
        super().__init__(embedding_dim, data_dim, gen_dims)
        self.categorical_start_idx = categorical_start_idx
        self.cat_counts = cat_counts
        self.predict_noise = predict_noise

    def forward(self, x):
        data = self.seq(x)

        if self.predict_noise:
            # Just predicting gaussian noise
            return data

        # Split into numerical and categorical outputs
        numerical_outputs = data[:, : self.categorical_start_idx]
        categorical_outputs = data[:, self.categorical_start_idx :]
        _idx = 0
        # Softmax over each category
        for k, v in self.cat_counts.items():
            categorical_outputs[:, _idx : _idx + v] = torch.softmax(categorical_outputs[:, _idx : _idx + v], dim=-1)
            _idx += v
        return torch.cat((numerical_outputs, categorical_outputs), dim=-1)
