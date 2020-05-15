"""Implements all the components of the DAGMM model."""

import torch
import numpy as np
from torch import nn

from compression_networks import CompressionNetwork
from estimation_networks import EstimationNetwork
from gmm import GMM
from utils import PyTorchDataset


class DAGMM(nn.Module):
    def __init__(self, input_size, comp_hiddens, comp_activation,
                 est_hiddens, est_activation, est_dropout_rate=0.5,
                 n_epoch=100, batch_size=128, learning_rate=1e-4,
                 lambda1=0.1, lambda2=0.005, device='cpu', 
                 verbose=True, random_seed=321):
        """
        Args:
            input_size: int
                the dimension of inputs.
            comp_hiddens: list of int
                sizes of hidden layers of compression network
                For example, if the sizes are [n1, n2],
                structure of compression network is:
                input_size -> n1 -> n2 -> n1 -> input_sizes
            comp_activation: function
                activation function of compression network
            est_hiddens: list of int
                sizes of hidden layers of estimation network.
                The last element of this list is assigned as n_comp.
                For example, if the sizes are [n1, n2],
                structure of estimation network is:
                input_size -> n1 -> n2 (= n_comp)
            est_activation: function
                activation function of estimation network
            est_dropout_rate: float (optional)
                dropout rate of estimation network applied during training
                if 0 or None, dropout is not applied.
            n_epoch: int (optional)
                epoch size during training, used when fit() called.
            batch_size: int (optional)
                mini-batch size during training
            learning_rate: float (optional)
                learning rate during training, used when fit() called.
            lambda1: float (optional)
                a parameter of loss function (for energy term)
            lambda2: float (optional)
                a parameter of loss function
                (for sum of diagonal elements of covariance)
            device: str (optional)
                'cpu' or 'cuda' ('cuda:x')
            verbose: bool (optional)
                print training loss when set to True.
            random_seed: int (optional)
                random seed, used when fit() called.
        """
        super().__init__()

        dimension_embedding = comp_hiddens[-1] + 2
        num_mixtures = est_hiddens[-1]
        self.compressor = CompressionNetwork(input_size, comp_hiddens,
                                             comp_activation)
        self.estimator = EstimationNetwork(dimension_embedding, est_hiddens,
                                           est_activation, est_dropout_rate)
        self.gmm = GMM(num_mixtures, dimension_embedding)
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.device = torch.device(device)
        self.verbose = verbose
        self.random_seed = random_seed
        self.eps = nn.Parameter(torch.Tensor([1e-6]), requires_grad=False)

    def forward(self, inputs):
        # Forward in the compression network.
        encoded = self.compressor.encode(inputs)
        decoded = self.compressor.decode(encoded)

        # Preparing the input for the estimation network.
        relative_ed = relative_euclidean_distance(inputs, decoded, self.eps)
        cosine_sim = cosine_similarity(inputs, decoded, self.eps)
        # Adding a dimension to prepare for concatenation.
        relative_ed = relative_ed.view(-1, 1)
        cosine_sim = relative_ed.view(-1, 1)
        z = torch.cat([encoded, relative_ed, cosine_sim], dim=1)
        # z has shape [batch_size, dim_embedding]

        # Updating the parameters of the mixture.
        if self.training:
            gamma = self.estimator(z)
            # gamma has shape [batch_size, num_mixtures]
            self.gmm._update_mixtures_parameters(z, gamma)
        # Estimating the energy of the samples.
        return self.gmm(z)

    def calc_loss(self, inputs):
        reconstruction_loss = self.compressor.reconstruction_loss(inputs)
        energy = torch.mean(self(inputs))
        penalty = self.gmm.cov_diag_loss()
        return reconstruction_loss + self.lambda1 * energy + self.lambda2 * penalty

    def fit(self, x):
        """Fit the DAGMM model according to the given data.

        Args:
            x: array-like, shape (n_samples, n_features)
                Training data.
        """
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        x = x.astype(np.float32)
        dataset = PyTorchDataset(x)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        self.to(self.device)
        self.train()

        for epoch in range(self.n_epoch):
            for x in dataloader:
                x = x.to(self.device)
                optimizer.zero_grad()
                loss = self.calc_loss(x)
                loss.backward()
                optimizer.step()

            if self.verbose:
                if (epoch + 1) % 100 == 0:
                    print(f"epoch {epoch+1}/{self.n_epoch} : loss = {loss.item():.3f}")

    def predict(self, x):
        """Calculate anormaly scores (sample energy) on samples in X.
        Args:
            x: array-like, shape (n_samples, n_features)
                Data for which anomaly scores are calculated.
                n_features must be equal to n_features of the fitted data.

        Returns:
            energies: array-like, shape (n_samples)
                Calculated sample energies.
        """
        x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        self.eval()
        if self.device == torch.device('cuda'):
            return self(x).detach().cpu().numpy()
        else:
            return self(x).detach().numpy()


def relative_euclidean_distance(x1, x2, eps):
    """x1 and x2 are assumed to be Variables or Tensors.
    They have shape [batch_size, dimension_embedding]"""
    num = torch.norm(x1 - x2, p=2, dim=1)  # dim [batch_size]
    denom = torch.norm(x1, p=2, dim=1)  # dim [batch_size]
    return num / torch.max(denom, eps)


def cosine_similarity(x1, x2, eps):
    """x1 and x2 are assumed to be Variables or Tensors.
    They have shape [batch_size, dimension_embedding]"""
    dot_prod = torch.sum(x1 * x2, dim=1)  # dim [batch_size]
    dist_x1 = torch.norm(x1, p=2, dim=1)  # dim [batch_size]
    dist_x2 = torch.norm(x2, p=2, dim=1)  # dim [batch_size]
    return dot_prod / torch.max(dist_x1*dist_x2, eps)
