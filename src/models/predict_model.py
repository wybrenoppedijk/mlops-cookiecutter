import glob
import logging

import hydra
import numpy as np
import torch
from PIL import Image

from src.models.model import MyAwesomeModel


@hydra.main(config_path="../../config", config_name="default_config.yaml")
def evaluate(config):
    logger = logging.getLogger(__name__)
    logger.info("Strat Evaluating..")
    # add any additional argument that you want
    hparam = config.predict
    model = MyAwesomeModel()
    model.load_state_dict(
        torch.load(hydra.utils.get_original_cwd() + "/" + hparam["model_path"])
    )
    model.eval()

    # load data
    images = hydra.utils.get_original_cwd() + "/" + hparam["data_path"]
    if images[-3:] == "npy" or images[-3:] == "npz" or images[-3:] == "pkl":
        eval_images = np.load(images).astype(np.float32)
        eval_images = torch.from_numpy(eval_images)
    elif images[-3:] == ".pt":
        eval_images = torch.load(images)
    else:
        filelist = glob.glob(images + "/*")
        eval_images = np.array([np.array(Image.open(fname)) for fname in filelist])

    # predict
    return model(eval_images[0].unsqueeze(1))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    evaluate()
