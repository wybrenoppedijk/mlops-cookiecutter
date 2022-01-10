# -*- coding: utf-8 -*-
import glob
import logging
import os
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    x = np.load(input_filepath + "/corruptmnist/test.npz", mmap_mode="r")

    images_test = torch.from_numpy(x.f.images).float()
    labels_test = torch.from_numpy(x.f.labels).long()

    try:
        os.makedirs(output_filepath + "/corruptmnist")
    except FileExistsError:
        # directory already exists
        pass
    torch.save(
        (images_test.unsqueeze(1), labels_test),
        output_filepath + "/corruptmnist/test.pt",
    )

    images_train = torch.Tensor()
    labels_train = torch.Tensor()
    for idx, file in enumerate(glob.glob(input_filepath + "/corruptmnist/train*.npz")):
        x = np.load(file, mmap_mode="r")
        images_train_part = torch.from_numpy(x.f.images).float()
        labels_train_part = torch.from_numpy(x.f.labels).long()
        images_train = torch.concat((images_train, images_train_part))
        labels_train = torch.concat((labels_train, labels_train_part))

    torch.save(
        (images_train.unsqueeze(1), labels_train.long()),
        output_filepath + "/corruptmnist/train.pt",
    )

    logger.info("Done!")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
