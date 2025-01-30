import os
import unittest
from unittest.mock import patch, MagicMock
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from data import CIFAR10Dataset

DATA_PATH = os.environ.get("DATA_PATH")


class TestCIFAR10Dataset(unittest.TestCase):
    @patch("torchvision.datasets.CIFAR10")
    def setUp(self, MockCIFAR10):
        # Mock the CIFAR10 dataset
        self.mock_train_data = MagicMock(spec=CIFAR10)
        self.mock_valid_data = MagicMock(spec=CIFAR10)
        MockCIFAR10.side_effect = [self.mock_train_data, self.mock_valid_data]

        # Mock the length of the dataset
        self.mock_train_data.__len__.return_value = 1000

        # Initialize the CIFAR10Dataset
        self.dataset = CIFAR10Dataset(dataset_path=DATA_PATH)

    def test_train_set_initialization(self):
        # Check if the train set is initialized correctly
        self.assertIsInstance(self.dataset.get_train_set(), Subset)
        self.assertEqual(len(self.dataset.get_train_set()), 950)  # 95% of 1000

    def test_valid_set_initialization(self):
        # Check if the valid set is initialized correctly
        self.assertIsInstance(self.dataset.get_valid_set(), Subset)
        self.assertEqual(len(self.dataset.get_valid_set()), 50)  # 5% of 1000

    def test_transformations(self):
        # Check if the transformations are applied correctly
        transform = self.dataset.transform
        transform_test = self.dataset.transform_test

        self.assertEqual(len(transform.transforms), 5)
        self.assertEqual(len(transform_test.transforms), 4)

    def test_dataloader(self):
        # Check if DataLoader works with the dataset
        train_loader = DataLoader(
            self.dataset.get_train_set(), batch_size=64, shuffle=True
        )
        valid_loader = DataLoader(
            self.dataset.get_valid_set(), batch_size=64, shuffle=False
        )

        self.assertEqual(len(train_loader.dataset), 950)
        self.assertEqual(len(valid_loader.dataset), 50)

        batch = next(iter(train_loader))
        print(batch)


if __name__ == "__main__":
    unittest.main()
