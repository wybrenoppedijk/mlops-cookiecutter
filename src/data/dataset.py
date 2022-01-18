import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class MNIST_Corrupted(Dataset):
    def __init__(self, data, train, transform=None):
        path = data + "/train.pt" if train else data + "/test.pt"
        self.data = torch.load(path)
        self.transform = transform

    def __len__(self):
        return self.data[0].size(0)

    def __getitem__(self, index) -> T_co:
        if self.transform is not None:
            return self.transform(self.data[0][index]), self.data[1][index]
        return self.data[0][index, :], self.data[1][index]
