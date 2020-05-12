"Defines the estimation networks."

import torch
from torch import nn
import torch.nn.functional as F


class EstimationNetwork(nn.Module):
    """Defines an estimation network.
    """
    def __init__(self, input_size, hidden_layer_sizes, activation=torch.tanh,
                 dropout_rate=0):
        """
        Args:
            input_size: int
                the dimension of inputs.
                typically the same as dimension_embedding.
            hidden_layer_sizes: list of int
                list of sizes of hidden layers.
                For example, if the sizes are [n1, n2],
                layer sizes of the network are:
                input_size -> n1 -> n2
                (network outputs the softmax probabilities of "n2" layer)
            activation: function
                activation function of hidden layer.
                the funtcion of last layer is softmax function.
            est_dropout_rate: float (optional)
                dropout rate of estimation network applied during training
                if 0 or None, dropout is not applied.
        """
        super().__init__()
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_rate)
        self.layers = nn.ModuleList()
        for i in range(len(hidden_layer_sizes)):
            if i == 0:
                self.layers.append(nn.Linear(input_size,
                                             hidden_layer_sizes[i]))
            else:
                self.layers.append(nn.Linear(hidden_layer_sizes[i-1],
                                             hidden_layer_sizes[i]))

    def forward(self, inputs):
        h = inputs
        for layer in self.layers[:-1]:
            h = self.dropout(self.activation(layer(h)))
        return F.softmax(self.layers[-1](h), dim=1)
