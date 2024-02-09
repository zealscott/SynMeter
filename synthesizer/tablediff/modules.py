"""
Code for the `TableDiffusion` model:
The first differentially-private diffusion model for tabular datasets.

https://arxiv.org/abs/2308.14784

@article{truda2023generating,
  title={Generating tabular datasets under differential privacy},
  author={Truda, Gianluca},
  journal={arXiv preprint arXiv:2308.14784},
  year={2023}
}
"""

import warnings

import numpy as np
import pandas as pd
import torch

from opacus import PrivacyEngine
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from .utils import calc_norm_dict
from .architectures import MixedTypeGenerator

# Ignore opacus hook warnings
warnings.filterwarnings(
    "ignore",
    message="Using a non-full backward hook when the forward contains multiple autograd Nodes",
)


# Function to compute the cosine noise schedule
def get_beta(t, T):
    return (1 - np.cos((np.pi * t) / T)) / 2 + 0.1


class TableDiffusion_Synthesiser(nn.Module):
    def __init__(
        self,
        batch_size=1024,
        lr=0.005,
        b1=0.5,
        b2=0.999,
        dims=(128, 128),
        diffusion_steps=5,
        predict_noise=True,
        max_grad_norm=1.0,
        epsilon_target=1.0,
        epoch_target=5,
        delta=1e-5,
        device="cuda:0",
    ):
        # Initialise parent (Generator) with the parameters
        super().__init__()
        self.device = torch.device(device)

        # Hyperparameters
        self.batch_size = batch_size
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.dims = dims
        self.diffusion_steps = diffusion_steps
        self.pred_noise = predict_noise
        self.max_grad_norm = max_grad_norm
        self.epoch_target = epoch_target

        # Setting privacy budget
        self.epsilon_target = epsilon_target
        self._delta = delta

        # Initialise training variables
        self._elapsed_batches = 0
        self._elapsed_epochs = 0
        self._eps = 0

    def fit(self, df, discrete_columns=[], verbose=True):
        self.data_dim = df.shape[1]
        self.data_n = df.shape[0]
        self.disc_columns = discrete_columns

        Tensor = torch.cuda.FloatTensor

        self.q_transformers = {}
        self.encoders = {}
        self.category_counts = {}

        # Preprocessing
        self._original_types = df.dtypes
        self._original_columns = df.columns
        df_encoded = df.select_dtypes(include="number").copy()  # numerical features
        df_encoded_cat = pd.DataFrame()  # categorical features
        for col in df.columns:
            if col in self.disc_columns:
                self.encoders[col] = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                transformed = self.encoders[col].fit_transform(df[col].values.reshape(-1, 1))
                transformed_df = pd.DataFrame(transformed, columns=[f"{col}_{i}" for i in range(transformed.shape[1])])
                df_encoded_cat = pd.concat([df_encoded_cat, transformed_df], axis=1)
                # Log the number of categories for each discrete column
                self.category_counts[col] = transformed_df.shape[1]
            else:
                self.q_transformers[col] = QuantileTransformer()
                df_encoded[col] = self.q_transformers[col].fit_transform(df[col].values.reshape(-1, 1))
                # check if there are any NaNs in the transformed data
        
        categorical_start_idx = df_encoded.shape[1]
        df_encoded = pd.concat([df_encoded, df_encoded_cat], axis=1)
        
        self.total_categories = sum(self.category_counts.values())
        self.encoded_columns = df_encoded.columns  # store the column names of the encoded data
        self.data_dim = df_encoded.shape[1]  # store the dimensionality of the encoded data
        self.data_n = df_encoded.shape[0]  # store the total number of data points

        # Convert df to tensor and wrap in DataLoader
        train_data = DataLoader(
            torch.from_numpy(df_encoded.values.astype(np.float32)).to(self.device),
            batch_size=self.batch_size,
            drop_last=False,
        )
        
        # check if there are any NaNs in the transformed data
        if torch.isnan(train_data.dataset).any():
            raise ValueError(f"NaNs found in transformed data")

        # Create MLP model
        self.model = MixedTypeGenerator(
            df_encoded.shape[1],
            self.data_dim,
            self.dims,
            self.pred_noise,
            categorical_start_idx,
            self.category_counts,
        ).to(self.device)

        # Initialise optimiser (and scheduler)
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=(self.b1, self.b2),
        )

        self.privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=False)
        self.model, self.optim, train_data = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optim,
            data_loader=train_data,
            target_epsilon=self.epsilon_target,
            target_delta=self._delta,
            epochs=self.epoch_target,
            max_grad_norm=self.max_grad_norm,
            poisson_sampling=True,
        )

        # Define loss functions
        mse_loss = nn.MSELoss()
        kl_loss = nn.KLDivLoss(reduction="batchmean")

        # Enforce training mode
        self.model.train()

        # Training loop
        for epoch in range(self.epoch_target):
            self._elapsed_epochs += 1
            for i, X in enumerate(train_data):
                # Check if loss is NaN and early stop
                if i > 2 and loss.isnan():
                    print("Loss is NaN. Early stopping.")
                    return self

                self._elapsed_batches += 1

                # Configure input
                real_X = Variable(X.type(Tensor))
                agg_loss = torch.Tensor([0]).to(self.device)

                # Diffusion process with cosine noise schedule
                for t in range(self.diffusion_steps):
                    self._eps = self.privacy_engine.get_epsilon(self._delta)
                    if self._eps >= self.epsilon_target:
                        print(f"Privacy budget reached in epoch {epoch} (batch {i}, {t=}).")
                        return self
                    beta_t = get_beta(t, self.diffusion_steps)
                    noise = torch.randn_like(real_X).to(self.device) * np.sqrt(beta_t)
                    noised_data = real_X + noise

                    if self.pred_noise:
                        # Use the model as a diffusion noise predictor
                        predicted_noise = self.model(noised_data)

                        # Calculate loss between predicted and actualy noise using MSE
                        numeric_loss = mse_loss(predicted_noise, noise)
                        categorical_loss = torch.tensor(0.0)
                        loss = numeric_loss
                    else:
                        # Use the model as a mixed-type denoiser
                        denoised_data = self.model(noised_data)

                        # Calculate numeric loss using MSE
                        if categorical_start_idx == 0:
                            numeric_loss = torch.tensor(0.0)
                        else:
                            numeric_loss = mse_loss(
                                denoised_data[:, :categorical_start_idx],
                                real_X[:, :categorical_start_idx],
                            )

                        # Convert categoricals to log-space (to avoid underflow issue) and calculate KL loss for each original feature
                        _idx = categorical_start_idx
                        categorical_losses = []
                        for _col, _cat_len in self.category_counts.items():
                            categorical_losses.append(
                                kl_loss(
                                    torch.log(denoised_data[:, _idx : _idx + _cat_len]),
                                    real_X[:, _idx : _idx + _cat_len],
                                )
                            )
                            _idx += _cat_len

                        # Average categorical losses over total number of categories across all categorical features
                        categorical_loss = sum(categorical_losses) / self.total_categories if categorical_losses else torch.tensor(0.0)

                        loss = numeric_loss + categorical_loss

                    # Add losses from each diffusion step
                    agg_loss += loss

                # Average loss over diffusion steps
                loss = agg_loss / self.diffusion_steps
                # print(f"Batches: {self._elapsed_batches}, {agg_loss=}")

                # Backward propagation and optimization step
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            print(
                f"[Epoch {epoch}/{self.epoch_target}] numerical loss: {numeric_loss.item():.6f}, categorical loss: {categorical_loss.item():.6f}, epsilon: {self._eps:.6f}"
            )

    def sample(self, n):
        self.model.eval()
        # Generate noise samples
        samples = torch.randn((n, self.data_dim)).to(self.device)

        # Generate synthetic data by runnin reverse diffusion process
        with torch.no_grad():
            for t in range(self.diffusion_steps - 1, -1, -1):
                beta_t = get_beta(t, self.diffusion_steps)
                noise_scale = np.sqrt(beta_t)
                print(f"Sampling {t=}, {np.sqrt(beta_t)=}")

                if self.pred_noise:
                    # Repeatedly predict and subtract noise
                    pred_noise = self.model(samples)
                    predicted_noise = pred_noise * noise_scale
                    samples = samples - predicted_noise
                else:
                    # Repeatedly denoise
                    samples = self.model(samples)

        synthetic_data = samples.detach().cpu().numpy()

        # Postprocessing: apply inverse transformations
        df_synthetic = pd.DataFrame(synthetic_data, columns=self.encoded_columns)
        for col in self.encoders:
            transformed_cols = [c for c in df_synthetic.columns if c.startswith(f"{col}_")]
            if transformed_cols:
                encoded_data = df_synthetic[transformed_cols].values
                df_synthetic[col] = self.encoders[col].inverse_transform(encoded_data).ravel()
                df_synthetic = df_synthetic.drop(columns=transformed_cols)

        for col in self.q_transformers:
            df_synthetic[col] = self.q_transformers[col].inverse_transform(df_synthetic[col].values.reshape(-1, 1))

        # Cast to the original datatypes for dataframe compatibility
        df_synthetic = df_synthetic.astype(self._original_types)
        # Order the columns as they were in the original dataframe
        df_synthetic = df_synthetic[self._original_columns]

        return df_synthetic
