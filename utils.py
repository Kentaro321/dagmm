import torch


class PyTorchDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = X

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx])

    def __len__(self):
        return len(self.X)
