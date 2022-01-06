import glob
import hydra
import torch

def mnist():
    test = torch.load(hydra.utils.get_original_cwd() + "/data/processed/corruptmnist/test.pt")
    train = []
    for file in glob.glob(hydra.utils.get_original_cwd() + "/data/processed/corruptmnist/train*.pt"):
        train.append(torch.load(file))

    return test, train
