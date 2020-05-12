"""Defines the compression network."""

import torch
from torch import nn


class CompressionNetwork(nn.Module):
    """Defines a compression network.
    """
    def __init__(self, input_size, hidden_layer_sizes, activation=torch.tanh):
        """
        Args:
            input_size: int
                the dimension of inputs.
            hidden_layer_sizes: list of int
                list of the size of hidden layers.
                For example, if the sizes are [n1, n2],
                the sizes of created networks are:
                input_size -> n1 -> n2 -> n1 -> input_sizes
                (network outputs the representation of "n2" layer)
            activation: function
                activation function of hidden layer.
                the last layer uses linear function.
        """
        super().__init__()
        self.activation = activation
        self._reconstruction_loss = nn.MSELoss()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(len(hidden_layer_sizes)):
            if i == 0:
                self.encoder.append(nn.Linear(input_size,
                                              hidden_layer_sizes[i]))
                self.decoder.append(nn.Linear(hidden_layer_sizes[i],
                                              input_size))
            else:
                self.encoder.append(nn.Linear(hidden_layer_sizes[i-1],
                                              hidden_layer_sizes[i]))
                self.decoder.append(nn.Linear(hidden_layer_sizes[i],
                                              hidden_layer_sizes[i-1]))
        self.decoder = self.decoder[::-1]  # reverse the order of layers

    def forward(self, inputs):
        out = self.encode(inputs)
        out = self.decode(out)
        return out

    def encode(self, inputs):
        h = inputs
        for layer in self.encoder[:-1]:
            h = self.activation(layer(h))
        return self.encoder[-1](h)

    def decode(self, inputs):
        h = inputs
        for layer in self.decoder[:-1]:
            h = self.activation(layer(h))
        return self.decoder[-1](h)

    def reconstruction_loss(self, inputs):
        reconstructed = self(inputs)
        return self._reconstruction_loss(inputs, reconstructed)
