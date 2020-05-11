"""Implements a GMM model."""

import numpy as np
import torch
from torch import nn


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

        self.Phi = torch.rand(num_mixtures)
        self.Phi = nn.Parameter(self.Phi, requires_grad=False)

        self.mu = torch.randn(num_mixtures, dimension_embedding)
        self.mu = nn.Parameter(self.mu, requires_grad=False)

        self.Sigma = torch.eye(dimension_embedding).repeat(num_mixtures, 1, 1)
        self.Sigma = nn.Parameter(self.Sigma, requires_grad=False)
        self.eps_Sigma = 1e-8 * torch.eye(dimension_embedding).repeat(num_mixtures, 1, 1)

    def forward(self, inputs):
        """Return the mean of energy.

        Args:
            inputs (Variable of shape [batch_size, dimension_embedding]):
                typically the input of the estimation network. The points
                in the embedding space.
        """
        dimension_embedding = inputs.shape[1]

        logphi = torch.log(self.Phi)  # [num_mixtures]
        coef = 0.5 * dimension_embedding * np.log(2 * np.pi)  # [1]
        logdet = 0.5 * torch.logdet(self.Sigma)  # [num_mixtures]

        diff = inputs[None] - self.mu[:, None]
        # diff has shape [num_mixtures, batch_size, dimension_embedding]
        mahala = 0.5 * torch.sum(torch.matmul(diff, self.Sigma) * diff, dim=2)
        # mahala has shape [num_mixtures, batch_size]

        result = logphi[:, None] - coef - logdet[:, None] - mahala
        result = torch.logsumexp(result, dim=0)
        return -torch.mean(result)

    def _update_mixtures_parameters(self, samples, mixtures_affiliations):
        """
        Args:
            samples (Variable of shape [batch_size, dimension_embedding]):
                typically the input of the estimation network. The points
                in the embedding space.
            mixtures_affiliations (Variable of shape [batch_size, num_mixtures])
                the probability of affiliation of each sample to each mixture.
                Typically the output of the estimation network.
        """
        if not self.training:
            # This function should not be used when we are in eval mode.
            return

        self.Phi.data = torch.mean(mixtures_affiliations, dim=0)

        mu_numer = torch.mm(mixtures_affiliations.T, samples)
        mu_denom = torch.sum(mixtures_affiliations, dim=0).view(-1, 1)
        self.mu.data = mu_numer / mu_denom

        diff = samples[None] - self.mu[:, None]
        # diff has shape [num_mixtures, batch_size, dimension_embedding]
        gamma = mixtures_affiliations.T[:, :, None]
        sigma_numer = torch.matmul(torch.transpose(diff * gamma, 1, 2), diff)
        self.Sigma.data = sigma_numer / mu_denom.view(-1, 1, 1) + self.eps_Sigma
