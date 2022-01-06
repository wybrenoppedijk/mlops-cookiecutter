import logging
import sys

import click
import hydra
import wandb
import numpy as np
import torch
import torch.nn.functional as F
from model import MyAwesomeModel

from src.data.dataset import mnist
from src.visualization.visualize_train import plot_loss
from src.visualization.visualize import Visuals


def acc(y_hat, y) -> float:
    _, predicted = torch.max(y_hat.data, 1)
    correct = (predicted == y.long()).sum().item()
    return correct / len(y)


@hydra.main(config_path="../../config", config_name='default_config.yaml')
def train(config):
    hparams = config.train
    wandb.init(
        # Set entity to specify your username or team name
        # ex: entity="carey",
        # Set the project where this run will be logged
        project="mlops-mnist",
        # Track hyperparameters and run metadata
        config=hparams)

    logger = logging.getLogger(__name__)
    logger.info("Strat Training..")

    # add any additional argument that you want
    model = MyAwesomeModel(config.model)
    losses, losses_val, acc_tests = [], [], []
    for epoch in range(hparams['epoch']):
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=hparams['lr'])
        test, train_set = mnist()
        loss_avg, loss_val_avg, acc_avg = [], [], []
        for batch in train_set:
            x, y = batch
            y_hat = model(x.unsqueeze(1))
            train_acc = acc(y_hat, y)
            # loss = F.cross_entropy(y_hat, y.long())
            loss = F.nll_loss(y_hat, y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_avg.append(loss.data.item())
            acc_avg.append(train_acc)
        losses.append(np.mean(loss_avg))
        acc_tests.append(np.mean(acc_avg))
        model.eval()

        y_val_hat = model(test[0].unsqueeze(1))
        losses_val.append(F.nll_loss(y_val_hat, test[1].long()))
        test_acc = acc(y_val_hat, test[1])

        wandb.log(
            {"acc_train": acc_tests[-1], "acc_tst": test_acc, "Loss_train": losses[-1], "loss_val": losses_val[-1]})

        if epoch % 2 == 0:
            viz = Visuals(model, torch.load(config.data['data_path'] + '/test.pt'))
            img=viz.intermediate_representation("conv.6")
            plt = viz.intermediate_distribution("conv.6")
            wandb.log({"Intermediate Representation": wandb.Image(img)})
            wandb.log({"Intermediate Distribution": wandb.Image(plt)})

        logger.info("Epoch: {}/{} Train Loss: {}".format(epoch, hparams['epoch'], losses[-1]))
        plot_loss(losses, losses_val, epoch)
        torch.save(model.state_dict(), hydra.utils.get_original_cwd() + "/models/corruptmnist/" + hparams['model_name'])


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    train()
