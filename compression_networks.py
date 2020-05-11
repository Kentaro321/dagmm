"""Defines the compression network."""

import torch
from torch import nn


class CompressionNetworkArrhythmia(nn.Module):
    """Defines a compression network for the Arrhythmia dataset as described in
    the paper."""
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(nn.Linear(274, 10),
                                     nn.Tanh(),
                                     nn.Linear(10, 2))
        self.decoder = nn.Sequential(nn.Linear(2, 10),
                                     nn.Tanh(),
                                     nn.Linear(10, 274))

        self._reconstruction_loss = nn.MSELoss()

    def forward(self, inputs):
        out = self.encoder(inputs)
        out = self.decoder(out)

        return out

    def encode(self,  inputs):
        return self.encoder(inputs)

    def decode(self, inputs):
        return self.decoder(inputs)

    def reconstruction_loss(self, inputs, target):
        target_hat = self(inputs)
        return self._reconstruction_loss(target_hat, target)
