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

        mixtures = [Mixture(dimension_embedding) for _ in range(num_mixtures)]
        self.mixtures = nn.ModuleList(mixtures)

    def forward(self, inputs):
        out = torch.zeros(inputs.shape[0])
        for mixture in self.mixtures:
            out += mixture(inputs)
        return -torch.sum(torch.log(out))

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

        for i, mixture in enumerate(self.mixtures):
            affiliations = mixtures_affiliations[:, i]
            mixture._update_parameters(samples, affiliations)



class Mixture(nn.Module):
    def __init__(self, dimension_embedding):
        super().__init__()
        self.dimension_embedding = dimension_embedding

        self.Phi = torch.rand(1)
        self.Phi = nn.Parameter(self.Phi, requires_grad=False)

        # Mu is the center/mean of the mixtures.
        self.mu = torch.randn(dimension_embedding)
        self.mu = nn.Parameter(self.mu, requires_grad=False)

        # Sigma encodes the shape of the gaussian 'bubble' of a given mixture.
        self.Sigma = torch.eye(dimension_embedding)
        self.Sigma = nn.Parameter(self.Sigma, requires_grad=False)

        # We'll use this to augment the diagonal of Sigma and make sure it is
        # inversible.
        self.eps_Sigma = 1e-8 * torch.eye(dimension_embedding)


    def forward(self, samples):
        """Samples has shape [batch_size, dimension_embedding]"""
        D = samples.shape[1]
        inv_sigma = torch.inverse(self.Sigma)
        det_sigma = torch.det(self.Sigma)
        det_sigma = torch.autograd.Variable(det_sigma)

        diff = samples - self.mu  # (B, D)
        out = -0.5 * torch.sum(torch.mm(diff, inv_sigma) * diff, dim=1)  # (B, )
        out = self.Phi * torch.exp(out) / torch.sqrt((2 * np.pi) ** D * det_sigma)
        return out

    def _update_parameters(self, samples, affiliations):
        """
        Args:
            samples (Variable of shape [batch_size, dimension_embedding]):
                typically the input of the estimation network. The points
                in the embedding space.
            mixtures_affiliations (Variable of shape [batch_size])
                the probability of affiliation of each sample to each mixture.
                Typically the output of the estimation network.
        """
        if not self.training:
            # This function should not be used when we are in eval mode.
            return

        # Updating phi.
        phi = torch.mean(affiliations)
        self.Phi.data = phi.data

        # Updating mu.
        mu = torch.sum(affiliations * samples, dim=0) / torch.sum(affiliations)
        self.mu.data = mu.data

        # Updating Sigma.
        diff = samples - mu  # (B, D)
        sigma = torch.mm(affiliations * diff.T, diff) / torch.sum(affiliations)
        self.Sigma.data = sigma.data * self.eps_Sigma
