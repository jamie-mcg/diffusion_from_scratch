import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Subset
import numpy as np


class CIFAR10Dataset:
    def __init__(
        self,
        dataset_path,
        crop_box=(25, 50, 25 + 128, 50 + 128),
        image_size=128,
        valid_split=0.05,
    ):
        self.dataset_path = dataset_path
        self.crop_box = crop_box
        self.image_size = image_size
        self.valid_split = valid_split

        self.transform = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        self.transform_test = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )

        self.train_set = torchvision.datasets.CIFAR10(
            self.dataset_path, train=True, transform=self.transform, download=True
        )
        self.valid_set = torchvision.datasets.CIFAR10(
            self.dataset_path,
            train=False,
            transform=self.transform_test,
            download=True,
        )

        self._create_train_valid_split()

    def _create_train_valid_split(self):
        num_train = len(self.train_set)
        indices = torch.randperm(num_train).tolist()
        valid_size = int(np.floor(self.valid_split * num_train))

        train_idx, valid_idx = indices[valid_size:], indices[:valid_size]

        self.train_set = Subset(self.train_set, train_idx)
        self.valid_set = Subset(self.valid_set, valid_idx)

    def get_train_set(self):
        return self.train_set

    def get_valid_set(self):
        return self.valid_set


class CelebADataset:
    def __init__(
        self,
        dataset_path,
        crop_box=(25, 50, 25 + 128, 50 + 128),
        image_size=128,
        valid_split=0.05,
    ):
        self.dataset_path = dataset_path
        self.crop_box = crop_box
        self.image_size = image_size
        self.valid_split = valid_split

        self.transform = T.Compose(
            [
                T.Lambda(lambda img: img.crop(self.crop_box)),
                T.Resize(self.image_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )

        self.transform_test = T.Compose(
            [
                T.Lambda(lambda img: img.crop(self.crop_box)),
                T.Resize(self.image_size),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )

        self.train_set = torchvision.datasets.CelebA(
            self.dataset_path, train=True, transform=self.transform, download=True
        )
        self.valid_set = torchvision.datasets.CelebA(
            self.dataset_path,
            train=False,
            transform=self.transform_test,
            download=True,
        )

        self._create_train_valid_split()

    def _create_train_valid_split(self):
        num_train = len(self.train_set)
        indices = torch.randperm(num_train).tolist()
        valid_size = int(np.floor(self.valid_split * num_train))

        train_idx, valid_idx = indices[valid_size:], indices[:valid_size]

        self.train_set = Subset(self.train_set, train_idx)
        self.valid_set = Subset(self.valid_set, valid_idx)

    def get_train_set(self):
        return self.train_set

    def get_valid_set(self):
        return self.valid_set


if __name__ == "__main__":
    import os
    import torch

    DATA_PATH = os.environ.get("DATA_PATH")

    dataset = CIFAR10Dataset(dataset_path=DATA_PATH)
