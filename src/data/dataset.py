import glob

import torch


def mnist():
    test = torch.load("data/processed/corruptmnist/test.pt")
    train = []
    for file in glob.glob("data/processed/corruptmnist/train*.pt"):
        train.append(torch.load(file))

    return test, train
