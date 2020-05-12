import numpy as np
import torch

from gmm import GMM
from model import DAGMMArrhythmia


def test_update_gmm():
    batch_size = 10
    dimension_embedding = 7
    num_mixtures = 2

    gmm = GMM(num_mixtures, dimension_embedding)

    latent_vectors = torch.randn(batch_size, dimension_embedding)
    affiliations = torch.nn.functional.softmax(torch.rand(batch_size, num_mixtures), dim=1)

    print('----------parameters before update----------')
    for param in gmm.parameters():
        print(param)

    gmm.train()
    gmm._update_mixtures_parameters(latent_vectors, affiliations)

    print('----------parameters after update----------')
    for param in gmm.parameters():
        print(param)


def test_forward_gmm():
    batch_size = 10
    dimension_embedding = 7
    num_mixtures = 2

    gmm = GMM(num_mixtures, dimension_embedding)
    latent_vectors = torch.randn(batch_size, dimension_embedding)

    gmm.train()
    out = gmm(latent_vectors)
    print(out)


def test_forward_dagmm():
    net = DAGMMArrhythmia()

    inputs = torch.rand(10, 274)
    out = net(inputs)
    print(out)


def test_calc_loss_dagmm():
    net = DAGMMArrhythmia()

    inputs = torch.rand(10, 274)
    loss = net.calc_loss(inputs)
    print(loss)


if __name__ == '__main__':
    # test_update_gmm()
    # test_forward_gmm()
    # test_forward_dagmm()
    test_calc_loss_dagmm()
