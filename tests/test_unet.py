import unittest
import torch
from unet import (
    PositionalEncoding,
    SelfAttention,
    DoubleConv,
    Down,
    Up,
    OutConv,
    UNet,
)


class TestPositionalEncoding(unittest.TestCase):
    def test_positional_encoding_shape(self):
        d_model = 64
        max_len = 5000
        pe = PositionalEncoding(d_model, max_len)
        x = torch.zeros(max_len, 1, d_model)
        out = pe(x)
        self.assertEqual(out.shape, (max_len, 1, d_model))

    def test_positional_encoding_values(self):
        d_model = 64
        max_len = 10
        pe = PositionalEncoding(d_model, max_len)
        x = torch.zeros(max_len, 1, d_model)
        out = pe(x)
        self.assertNotEqual(
            out.sum().item(), 0
        )  # Ensure that positional encoding adds non-zero values


class TestSelfAttention(unittest.TestCase):
    def test_self_attention_shape(self):
        in_channels = 64
        sa = SelfAttention(in_channels)
        x = torch.randn(1, in_channels, 32, 32)
        out = sa(x)
        self.assertEqual(out.shape, x.shape)

    def test_self_attention_values(self):
        in_channels = 64
        sa = SelfAttention(in_channels)
        x = torch.randn(1, in_channels, 32, 32)
        out = sa(x)
        self.assertNotEqual(
            out.sum().item(), 0
        )  # Ensure that self-attention modifies the input


class TestUNet(unittest.TestCase):
    def setUp(self):
        self.model = UNet(n_channels=3, n_classes=1)
        self.input_tensor = torch.randn(1, 3, 256, 256)

    def test_unet_forward_shape(self):
        output = self.model(self.input_tensor)
        self.assertEqual(output.shape, (1, 1, 256, 256))

    def test_unet_forward_values(self):
        output = self.model(self.input_tensor)
        self.assertNotEqual(
            output.sum().item(), 0
        )  # Ensure that the model produces non-zero output

    def test_unet_positional_encoding(self):
        x1 = self.model.inc(self.input_tensor)
        x1_pe = self.model.positional_encoding(x1)
        self.assertNotEqual(
            x1_pe.sum().item(), x1.sum().item()
        )  # Ensure positional encoding modifies the input

    def test_unet_self_attention(self):
        x1 = self.model.inc(self.input_tensor)
        x1_pe = self.model.positional_encoding(x1)
        x1_sa = self.model.self_attention(x1_pe)
        self.assertNotEqual(
            x1_sa.sum().item(), x1_pe.sum().item()
        )  # Ensure self-attention modifies the input


if __name__ == "__main__":
    unittest.main()
