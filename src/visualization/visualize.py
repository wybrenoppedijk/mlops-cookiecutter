from typing import Callable, Dict, Iterable

import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from torch import Tensor, nn


class FeatureExtractor(nn.Module):
    """
    FeatureExtractor is a nn.Module that extracts features from a layer in a model.

    Attributes:
        model: The model to extract features from.
        layer: The layer to extract features from.
    Methods:
        forward: Returns the features of the layer in the model.
        save_features: Saves the features of the layer in the model to a file.
    """

    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output

        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._features


class Visuals(object):
    def __init__(self, model, data):
        self.model = model
        self.model.eval()
        self.data = data
        self.features = {}

    def intermediate_representation(self, layer_id, epoch):
        """Plots the intermediate representation of the model for a given layer.
        Parameters:
                data (tensor): a 4 dimensional tensor of shape (batch_size, channels, height, width)
                layer_id (str): the name of the layer to visualize
                dim_reduction (bool): whether to reduce the dimensionality of the feature map

        Returns:
                None
        """
        plt.clf()
        model_features = FeatureExtractor(self.model, layers=[layer_id])
        images, y = self.data
        activations = model_features(images)[layer_id]
        img = 1
        activations = activations[img, :]
        n_features = activations.shape[0]
        images_per_row = 16
        size = activations.shape[-1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = activations[col * images_per_row + row - 1, :, :]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image.detach().numpy(), 0, 255).astype(
                    "uint8"
                )
                display_grid[
                    col * size : (col + 1) * size, row * size : (row + 1) * size
                ] = channel_image
        scale = 1.0 / size
        plt.figure(
            figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0])
        )
        plt.imshow(display_grid, aspect="auto", cmap="viridis")
        plt.title(layer_id)
        plt.grid(False)
        plt.savefig(
            f"{hydra.utils.get_original_cwd()}"
            f"/reports/figures/representation-{layer_id}-ep-{epoch}.png"
        )
        return plt

    def intermediate_distribution(self, layer_id, epoch):
        """Plots the intermediate representation of the model for a given layer.
        Parameters:
                data (tensor): a 4 dimensional tensor of shape (batch_size, channels, height, width)
                layer_id (str): the name of the layer to visualize
                dim_reduction (bool): whether to reduce the dimensionality of the feature map

        Returns:
                None
        """
        plt.clf()
        model_features = FeatureExtractor(self.model, layers=[layer_id])
        images, y = self.data
        activations = model_features(images)[layer_id]
        tsne = TSNE(n_components=2, init="pca", random_state=0, learning_rate="auto")
        activations = tsne.fit_transform(
            activations.detach().numpy().reshape(activations.size(0), -1)
        )
        plt.figure(figsize=(8, 8))
        sns.scatterplot(
            x=activations[:, 0],
            y=activations[:, 1],
            hue=y.data.numpy(),
            palette=sns.color_palette("hls", 10),
            legend="full",
            alpha=0.3,
        )
        plt.title("t-SNE visualization of layer {}".format(layer_id))
        plt.savefig(
            f"{hydra.utils.get_original_cwd()}"
            f"/reports/figures/distribution-{layer_id}-ep-{epoch}.png"
        )
        return plt
