import logging
import sys

import click
import numpy as np
import torch
import torch.nn.functional as F
from model import MyAwesomeModel

from src.data.dataset import mnist
from src.visualization.visualize_train import plot_loss


@click.command()
@click.argument("lr", type=click.FLOAT, default=0.001)
def train(lr):
    logger = logging.getLogger(__name__)
    logger.info("Strat Training..")
    # add any additional argument that you want
    model = MyAwesomeModel()
    losses, losses_val = [], []
    for epoch in range(10):
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        test, train_set = mnist()
        loss_avg, loss_val_avg = [], []
        for batch in train_set:
            x, y = batch
            y_hat = model(x.unsqueeze(1))
            # loss = F.cross_entropy(y_hat, y.long())
            loss = F.nll_loss(y_hat, y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_avg.append(loss.data.item())
        losses.append(np.mean(loss_avg))
        model.eval()
        y_val_hat = model(test[0].unsqueeze(1))
        losses_val.append(F.nll_loss(y_val_hat, test[1].long()))
        logger.info("Epoch: {}/{} Train Loss: {}".format(epoch, 10, losses[-1]))
        plot_loss(losses, losses_val, epoch)
        torch.save(model.state_dict(), "models/mnistcorrupted/model.pt")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    train()
