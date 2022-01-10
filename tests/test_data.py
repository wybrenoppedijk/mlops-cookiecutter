

from src.data.dataset import MNIST_Corrupted

# Write test that tests the size of the test, train for the corrupted mnist dataset


def dataset_len():
    num_train = 40000
    num_test = 5000
    test = MNIST_Corrupted("data/processed/corruptmnist", train=False)
    train = MNIST_Corrupted("data/processed/corruptmnist", train=True)

    assert len(test) == num_test
    assert len(train) == num_train
