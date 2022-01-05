from typing import Callable, Dict, Iterable

import click
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from torch import Tensor, nn

from src.models.model import MyAwesomeModel


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
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.features = {}

    def intermediate_representation(self, data, layer_id, dim_reduction=True):
        """Plots the intermediate representation of the model for a given layer.
        Parameters:
                data (tensor): a 4 dimensional tensor of shape (batch_size, channels, height, width)
                layer_id (str): the name of the layer to visualize
                dim_reduction (bool): whether to reduce the dimensionality of the feature map

        Returns:
                None
        """
        model_features = FeatureExtractor(self.model, layers=[layer_id])
        images, y = data
        activations = model_features(images.unsqueeze(1))[layer_id]
        idx = np.random.randint(0, activations.shape[0], 1)
        activations_sample = activations[idx, :][0]
        n_features = activations_sample.shape[0]
        images_per_row = 16
        size = activations_sample.shape[-1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        if dim_reduction:
            tsne = TSNE(n_components=2)
            activations = tsne.fit_transform(
                activations.detach().numpy().reshape(activations.size(0), -1)
            )

            sns.scatterplot(
                x=activations[:, 0],
                y=activations[:, 1],
                hue=y.data.numpy(),
                palette=sns.color_palette("hls", 10),
                legend="full",
                alpha=0.3,
            )
            plt.title("t-SNE visualization of layer {}".format(layer_id))
        else:
            for col in range(n_cols):
                for row in range(images_per_row):
                    channel_image = activations_sample[
                        col * images_per_row + row - 1 :, :
                    ]
                    channel_image -= channel_image.mean()
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(
                        channel_image.detach().numpy(), 0, 255
                    ).astype("uint8")
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
        plt.savefig(f"reports/figures/{layer_id}.png")
        plt.clf()

    # Visualize features in a 2D space using t-SNE to do the dimensionality reduction


if __name__ == "__main__":
    model = MyAwesomeModel()
    model.load_state_dict(torch.load("models/mnistcorrupted/model.pt"))
    viz = Visuals(model)
    viz.intermediate_representation(
        torch.load("data/processed/corruptmnist/test.pt"), "conv1"
    )
