import os

import numpy as np
import pytest
import torch
from hydra import compose, initialize

from src.models.model import MyAwesomeModel

# write test that given input with shape X that the output of the model have shape Y


def test_model_shape():
    with initialize(config_path="../config"):
        # config is relative to a module
        config = compose(config_name="default_config.yaml")
        model = MyAwesomeModel(config)
        X = torch.rand(1, 1, 28, 28)
        Y = model.forward(X)
        assert Y.shape == (1, 10)


def test_model_forward_shape_warnings():
    with initialize(config_path="../config"):
        # config is relative to a module
        config = compose(config_name="default_config.yaml")
        model = MyAwesomeModel(config)
        with pytest.raises(ValueError, match="Expected 4D tensor as input, got 3D"):
            X = torch.rand(1, 28, 28)
            Y = model.forward(X)


def test_model_forward_channel_warnings():
    with initialize(config_path="../config"):
        # config is relative to a module
        config = compose(config_name="default_config.yaml")
        model = MyAwesomeModel(config)
        with pytest.raises(ValueError, match="Expected input with 1 channels, got 3"):
            X = torch.rand(1, 3, 28, 28)
            Y = model.forward(X)
