"""TVAE module."""

import numpy as np
import pandas as pd
import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .data_transformer import DataTransformerGMM
from .base import BaseSynthesizer, random_state


class Encoder(Module):
    """Encoder for the TVAE.

    Args:
        data_dim (int):
            Dimensions of the data.
        compress_dims (tuple or list of ints):
            Size of each hidden layer.
        embedding_dim (int):
            Size of the output vector.
    """

    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, input_):
        """Encode the passed `input_`."""
        feature = self.seq(input_)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class Decoder(Module):
    """Decoder for the TVAE.

    Args:
        embedding_dim (int):
            Size of the input vector.
        decompress_dims (tuple or list of ints):
            Size of each hidden layer.
        data_dim (int):
            Dimensions of the data.
    """

    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input_):
        """Decode the passed `input_`."""
        return self.seq(input_), self.sigma  # sigma is the standard deviation for continous columns


def _loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor):
    st = 0
    loss = []
    for column_info in output_info:
        for span_info in column_info:
            if span_info.activation_fn != "softmax":
                # continous columns
                ed = st + span_info.dim
                std = sigmas[st]
                eq = x[:, st] - torch.tanh(recon_x[:, st])
                loss.append((eq**2 / 2 / (std**2)).sum())
                loss.append(torch.log(std) * x.size()[0])
                st = ed

            else:
                ed = st + span_info.dim
                loss.append(
                    cross_entropy(recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction="sum")
                )
                st = ed

    assert st == recon_x.size()[1]
    # prior matching term KL[q(z|r)||p(z)]
    # mu, logvar, std is the parameters of q(z|r)
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]


class TVAE(Module):
    """TVAE."""

    def __init__(
        self,
        embedding_dim=128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        l2scale=1e-5,
        batch_size=500,
        epochs=300,
        loss_factor=2,
        cuda=None,
    ):
        super(TVAE, self).__init__()
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.epochs = epochs
        self.loss_history = pd.DataFrame(columns=["Epoch", "Batch", "loss"])

        self._device = cuda

    def fit(self, train_data, discrete_columns=()):
        """Fit the TVAE Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self.transformer = DataTransformerGMM()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)
        dataset = TensorDataset(torch.from_numpy(train_data.astype("float32")).to(self._device))
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        data_dim = self.transformer.output_dimensions
        encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)
        optimizerAE = Adam(
            list(encoder.parameters()) + list(self.decoder.parameters()), weight_decay=self.l2scale
        )

        epoch_iterator = tqdm(range(self.epochs))
        for i in epoch_iterator:
            for id_, data in enumerate(loader):
                optimizerAE.zero_grad()
                real = data[0].to(self._device)
                mu, std, logvar = encoder(real)  # from r to z
                eps = torch.randn_like(std)
                emb = eps * std + mu  # reparameterization trick to get random z
                rec, sigmas = self.decoder(emb)  # from z to r
                loss_1, loss_2 = _loss_function(
                    rec, real, sigmas, mu, logvar, self.transformer.output_info_list, self.loss_factor
                )
                # loss_1 is the reconstruction term
                # loss_2 is the prior matching term KL[q(z|r)||p(z)]
                loss = loss_1 + loss_2
                self.loss_history.loc[len(self.loss_history)] = [i, id_, loss.item()]

                loss.backward()
                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)

    def sample(self, samples):
        """Sample data similar to the training data.

        Args:
            samples (int):
                Number of rows to sample.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        self.decoder.eval()

        steps = samples // self.batch_size + 1
        data = []
        for _ in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self._device)
            fake, sigmas = self.decoder(noise)
            fake = torch.tanh(fake)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:samples]
        return self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        self.decoder.to(self._device)

    
    #### the following code is for auditing the training process of TVAE model ####
    def auditing_fit(self, train_data, discrete_columns=(),interval = 10, store_dir= None, n_sample = 100):
        self.transformer = DataTransformerGMM()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)
        dataset = TensorDataset(torch.from_numpy(train_data.astype("float32")).to(self._device))
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        data_dim = self.transformer.output_dimensions
        encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)
        optimizerAE = Adam(
            list(encoder.parameters()) + list(self.decoder.parameters()), weight_decay=self.l2scale
        )

        epoch_iterator = tqdm(range(self.epochs))
        saved_interval = self.epochs // interval
        for i in epoch_iterator:
            if (i+1) % saved_interval == 0 or i == 0:
                # sample 
                # sample 
                sampled = self.sample(n_sample)
                import os
                os.makedirs(store_dir, exist_ok=True)
                sample_path = os.path.join(store_dir, 'gen_data_{}.csv'.format(i+1 if i > 0 else 0))
                sampled.to_csv(sample_path, index=False)
                print("save the sample data at {}".format(sample_path))
            for id_, data in enumerate(loader):
                optimizerAE.zero_grad()
                real = data[0].to(self._device)
                mu, std, logvar = encoder(real)  # from r to z
                eps = torch.randn_like(std)
                emb = eps * std + mu  # reparameterization trick to get random z
                rec, sigmas = self.decoder(emb)  # from z to r
                loss_1, loss_2 = _loss_function(
                    rec, real, sigmas, mu, logvar, self.transformer.output_info_list, self.loss_factor
                )
                # loss_1 is the reconstruction term
                # loss_2 is the prior matching term KL[q(z|r)||p(z)]
                loss = loss_1 + loss_2
                self.loss_history.loc[len(self.loss_history)] = [i, id_, loss.item()]

                loss.backward()
                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)