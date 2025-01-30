import unittest
import torch
from unittest.mock import MagicMock

from ddpm import DDPM


class TestDDPM(unittest.TestCase):
    def setUp(self):
        self.img_size = 32
        self.n_steps = 1000
        self.beta_init = 1e-4
        self.beta_final = 1e-2
        self.schedule = "linear"
        self.ddpm = DDPM(
            img_size=self.img_size,
            n_steps=self.n_steps,
            beta_init=self.beta_init,
            beta_final=self.beta_final,
            schedule=self.schedule,
        )
        self.model = MagicMock()
        self.model.device = torch.device("cpu")
        self.model.return_value = torch.randn(1, 3, self.img_size, self.img_size)

    def test_get_noised_image(self):
        # Create a dummy image tensor
        x = torch.randn(1, 3, self.img_size, self.img_size)
        t = torch.randint(0, self.n_steps, (1,)).item()

        # Get the noised image
        noised_image = self.ddpm.get_noised_image(x, t)

        # Check the shape of the noised image
        self.assertEqual(noised_image.shape, x.shape)

        # Check that the noised image is a tensor
        self.assertIsInstance(noised_image, torch.Tensor)

        # Check that the noised image is not equal to the original image
        self.assertFalse(torch.equal(noised_image, x))

        # Check that the noised image is within the expected range
        self.assertTrue(torch.all(noised_image >= -1))
        self.assertTrue(torch.all(noised_image <= 1))

    def test_sample(self):
        n = 1
        samples = self.ddpm.sample(self.model, n)

        # Check the shape of the output
        self.assertEqual(samples.shape, (n, 3, self.img_size, self.img_size))

        # Check the type of the output
        self.assertIsInstance(samples, torch.Tensor)

        # Check the dtype of the output
        self.assertEqual(samples.dtype, torch.uint8)

        # Check the value range of the output
        self.assertTrue(torch.all(samples >= 0))
        self.assertTrue(torch.all(samples <= 255))


if __name__ == "__main__":
    unittest.main()
