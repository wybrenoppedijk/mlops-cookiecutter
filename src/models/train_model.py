import logging
import sys

import click
import hydra
import wandb
import numpy as np
import torch
import torch.nn.functional as F
from model import MyAwesomeModel
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.data.dataset import MNIST_Corrupted
from src.visualization.visualize_train import plot_loss
from src.visualization.visualize import Visuals


def acc(y_hat, y) -> float:
    _, predicted = torch.max(y_hat.data, 1)
    correct = (predicted == y.long()).sum().item()
    return correct / len(y)


@hydra.main(config_path="../../config", config_name='default_config.yaml')
def train(config):
    logger = logging.getLogger(__name__)
    logger.info("Strat Training..")

    mnist_train = DataLoader(MNIST_Corrupted(config.data['data_path'], train=True), batch_size=62, shuffle=True)
    mnist_val_dataset = MNIST_Corrupted(config.data['data_path'], train=False)
    mnist_val = DataLoader(mnist_val_dataset, batch_size=len(mnist_val_dataset))

    trainer = Trainer(max_epochs=10, logger=pl.loggers.WandbLogger(project="mlops-mnist", config=config))
    model = MyAwesomeModel(config)
    trainer.fit(model, mnist_train, mnist_val)

    # # add any additional argument that you want
    # model = MyAwesomeModel(config.model)
    # losses, losses_val, acc_tests = [], [], []
    # for epoch in range(hparams['epoch']):
    #     model.train()
    #     optimizer = torch.optim.AdamW(model.parameters(), lr=hparams['lr'])
    #     test, train_set = mnist()
    #     loss_avg, loss_val_avg, acc_avg = [], [], []
    #     for batch in train_set:
    #         x, y = batch
    #         y_hat = model(x.unsqueeze(1))
    #         train_acc = acc(y_hat, y)
    #         # loss = F.cross_entropy(y_hat, y.long())
    #         loss = F.nll_loss(y_hat, y.long())
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         loss_avg.append(loss.data.item())
    #         acc_avg.append(train_acc)
    #     losses.append(np.mean(loss_avg))
    #     acc_tests.append(np.mean(acc_avg))
    #     model.eval()
    #
    #     y_val_hat = model(test[0].unsqueeze(1))
    #     losses_val.append(F.nll_loss(y_val_hat, test[1].long()))
    #     test_acc = acc(y_val_hat, test[1])
    #
    #     wandb.log(
    #         {"acc_train": acc_tests[-1], "acc_tst": test_acc, "Loss_train": losses[-1], "loss_val": losses_val[-1]})
    #
    #     if epoch % 2 == 0:
    #         viz = Visuals(model, torch.load(config.data['data_path'] + '/test.pt'))
    #         plt2=viz.intermediate_representation("conv.6")
    #         wandb.log({"Intermediate Representation": wandb.Image(plt2)})
    #         plt1 = viz.intermediate_distribution("conv.6")
    #         wandb.log({"Intermediate Distribution": wandb.Image(plt1)})
    #
    #     logger.info("Epoch: {}/{} Train Loss: {}".format(epoch, hparams['epoch'], losses[-1]))
    #     plot_loss(losses, losses_val, epoch)
    #     torch.save(model.state_dict(), hydra.utils.get_original_cwd() + "/models/corruptmnist/" + hparams['model_name'])


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    train()
