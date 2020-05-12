"""Implements a GMM model."""

import numpy as np
import torch
from torch import nn

EPS = 1e-6


class GMM(nn.Module):
    """Implements a Gaussian Mixture Model."""
    def __init__(self, num_mixtures, dimension_embedding):
        """Creates a Gaussian Mixture Model.

        Args:
            num_mixtures (int): the number of mixtures the model should have.
            dimension_embedding (int): the number of dimension of the embedding
                space (can also be thought as the input dimension of the model)
        """
        super().__init__()
        self.num_mixtures = num_mixtures
        self.dimension_embedding = dimension_embedding

        self.phi = torch.zeros(num_mixtures)
        self.phi = nn.Parameter(self.phi, requires_grad=False)
        self.mu = torch.zeros(num_mixtures, dimension_embedding)
        self.mu = nn.Parameter(self.mu, requires_grad=False)
        self.sigma = torch.eye(dimension_embedding).repeat(num_mixtures, 1, 1)
        self.sigma = nn.Parameter(self.sigma, requires_grad=False)
        self.eps_sigma = EPS * torch.eye(dimension_embedding).repeat(num_mixtures, 1, 1)
        self.eps_sigma = nn.Parameter(self.eps_sigma, requires_grad=False)
        self.L = torch.cholesky(self.sigma + self.eps_sigma)
        self.L = nn.Parameter(self.L, requires_grad=False)
        # L is cholesky decomposition of self.sigma: LL^T = self.sigma

    def forward(self, z):
        """Return the mean of energy.

        Args:
            z (Variable of shape [batch_size, dimension_embedding]):
                typically the input of the estimation network. The points
                in the embedding space.
        """
        logphi = torch.log(self.phi[:, None])
        coef = 0.5 * self.dimension_embedding * np.log(2 * np.pi)
        logdet = 2.0 * torch.sum(torch.log(torch.diagonal(self.L, dim1=1, dim2=2)), dim=1)[:, None]

        diff = z[None] - self.mu[:, None]
        # diff has shape [num_mixtures, batch_size, dimension_embedding]
        diff = torch.triangular_solve(torch.transpose(diff, 1, 2), self.L).solution
        # diff has shape [num_mixtures, dimension_embedding, batch_size]
        mahala = 0.5 * torch.sum(diff**2, dim=1)
        # mahala has shape [num_mixtures, batch_size]

        result = logphi - coef - logdet - mahala
        return -torch.logsumexp(result, dim=0)

    def _update_mixtures_parameters(self, z, gamma):
        """
        Args:
            z (Variable of shape [batch_size, dimension_embedding]):
                typically the input of the estimation network. The points
                in the embedding space.
            gamma (Variable of shape [batch_size, num_mixtures])
                the probability of affiliation of each sample to each mixture.
                Typically the output of the estimation network.
        """
        if not self.training:
            # This function should not be used when we are in eval mode.
            return

        gamma_sum = torch.sum(gamma, dim=0)
        self.phi.data = torch.mean(gamma, dim=0)

        self.mu.data = torch.einsum('ik,il->kl', gamma, z) / gamma_sum[:, None]

        diff = torch.sqrt(gamma[:, :, None]) * (z[:, None] - self.mu[None])
        # diff has shape [batch_size, num_mixtures, dimension_embedding]
        self.sigma.data = torch.einsum('ikl,ikm->klm', diff, diff) / gamma_sum[:, None, None]
        self.sigma.data += self.eps_sigma.data

        self.L.data = torch.cholesky(self.sigma)

    def cov_diag_loss(self):
        return torch.sum(1. / torch.diagonal(self.sigma, dim1=1, dim2=2))
