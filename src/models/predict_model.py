import sys
import click
import logging

import torch
import numpy as np

from src.data.dataset import mnist
from model import MyAwesomeModel
from src.visualization.visualize_train import plot_loss
import torch.nn.functional as F
import glob
from PIL import Image

@click.command()
@click.argument('model', type=click.Path(exists=True))
@click.argument('images', type=click.Path(exists=True))
def evaluate(model, images):
    logger = logging.getLogger(__name__)
    logger.info('Strat Evaluating..')
    # add any additional argument that you want
    model = torch.load(model)
    model.eval()

    # load data]
    eval_images = None
    if images[-3:] == 'npy' or images[-3:] == 'npz' or images[-3:] == 'pkl':
        eval_images = np.load(images).astype(np.float32)
        eval_images = torch.from_numpy(images)
    else:
        filelist = glob.glob(images + '/*')
        eval_images = np.array([np.array(Image.open(fname)) for fname in filelist])


    # predict
    return model(eval_images.unsqueeze(1))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    evaluate()