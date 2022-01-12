"""
LFW dataloading
"""
import argparse
import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        # TODO: fill out with what you need
        self.dir_path = path_to_folder
        self.labels = []
        self.img_paths = []
        self.transform = transform
        self.get_imgs()

    def get_imgs(self):
        for label in os.listdir(self.dir_path):
            for file in glob.glob(self.dir_path + "/" + label + "/*.jpg"):
                self.img_paths.append(file)
                self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int) -> torch.Tensor:
        # TODO: fill out
        return self.transform(Image.open(self.img_paths[index]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-path_to_folder", default="", type=str)
    parser.add_argument("-num_workers", default=None, type=int)
    parser.add_argument("-visualize_batch", action="store_true")
    parser.add_argument("-get_timing", action="store_true")
    parser.add_argument("-errorplot", action="store_true")
    args = parser.parse_args()

    lfw_trans = transforms.Compose(
        [transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)), transforms.ToTensor()]
    )

    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)

    # Define dataloader
    # Note we need a high batch size to see an effect of using many
    # number of workers
    if args.errorplot:
        workers = [1, 2, 3, 4, 5]
        results = []
        stds = []
        for worker in workers:
            dataloader = DataLoader(dataset, batch_size=512, num_workers=worker)
            res = []
            for _ in range(3):
                start = time.time()
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx > 100:
                        break
                end = time.time()

                res.append(end - start)

            results.append(np.mean(res))
            stds.append(np.std(res))
            print(f"Timing: {np.mean(res)}+-{np.std(res)}")

        plt.errorbar(workers, results, yerr=stds, label="both limits (default)")
    else:
        dataloader = DataLoader(
            dataset, batch_size=512, shuffle=False, num_workers=args.num_workers
        )

    if args.visualize_batch:
        figure = plt.figure(figsize=(8, 8))
        cols, rows = 3, 3
        imgs = next(iter(dataloader))
        for i in range(1, cols * rows + 1):
            sample_idx = torch.randint(len(imgs), size=(1,)).item()
            img = imgs[sample_idx]
            figure.add_subplot(rows, cols, i)
            # plt.title(dataloader[label])
            plt.axis("off")
            plt.imshow(np.transpose(img.numpy(), (1, 2, 0)), cmap="gray")
        plt.show()

    if args.get_timing:
        # lets do so repetitions
        res = []
        for _ in range(5):
            start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > 100:
                    break
            end = time.time()

            res.append(end - start)

        res = np.array(res)
        print(f"Timing: {np.mean(res)}+-{np.std(res)}")
