import logging

import hydra
import pytorch_lightning as pl
import torch
from model import MyAwesomeModel
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from src.data.dataset import MNIST_Corrupted


def acc(y_hat, y) -> float:
    _, predicted = torch.max(y_hat.data, 1)
    correct = (predicted == y.long()).sum().item()
    return correct / len(y)


@hydra.main(config_path="../../config", config_name="default_config.yaml")
def train(config):
    logger = logging.getLogger(__name__)
    logger.info("Strat Training..")

    mnist_train = DataLoader(
        MNIST_Corrupted(config.data["data_path"], train=True),
        batch_size=62,
        shuffle=True,
    )
    mnist_val_dataset = MNIST_Corrupted(config.data["data_path"], train=False)
    mnist_val = DataLoader(mnist_val_dataset, batch_size=len(mnist_val_dataset))

    trainer = Trainer(
        max_epochs=10,
        logger=pl.loggers.WandbLogger(project="mlops-mnist", config=config),
    )
    model = MyAwesomeModel(config)
    trainer.fit(model, mnist_train, mnist_val)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    train()
