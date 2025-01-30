import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, dim, n=10000):
        super(PositionalEncoding, self).__init__()
        self.dim = dim
        self.div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(n)) / dim)
        )

    def _calculate_pe(self, t):
        position = t.repeat(1, self.dim // 2)

        pe_even = torch.sin(position * self.div_term)
        pe_odd = torch.cos(position * self.div_term)

        # Combine even and odd terms
        pe = torch.zeros(position.size(0), self.dim)
        pe[:, 0::2] = pe_even
        pe[:, 1::2] = pe_odd

        return pe

    def forward(self, t):
        return self._calculate_pe(t)


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.atten = nn.MultiheadAttention(in_channels, 8, batch_first=True)
        self.norm = nn.LayerNorm([in_channels])
        self.ffn = nn.Sequential(
            nn.LayerNorm([in_channels]),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels),
        )

    def forward(self, x):
        # Reshape from [b, c, d1, d2] -> [b, d1*d2, c]
        d1, d2 = x.shape[2:]
        x = x.view(-1, self.in_channels, d1 * d2).transpose(1, 2)
        x_norm = self.norm(x)
        x = x + self.atten(x_norm, x_norm, x_norm)[0]
        x = x + self.ffn(x)
        return x.transpose(2, 1).view(-1, self.in_channels, d1, d2)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.second_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.first_conv(x)
        return self.second_conv(x1)


class Down(nn.Module):
    """Downscaling with maxpool then double conv and time embedding"""

    def __init__(self, in_channels, out_channels, time_dim):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels),
            DoubleConv(in_channels, out_channels),
        )

        self.time_emb = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))

    def forward(self, x, t):
        # Get the time embedding and broadcast it to the size of the input
        t_emb = self.time_emb(t)[:, :, None, None].repeat(
            1, 1, x.shape[-2] // 2, x.shape[-1] // 2
        )
        return self.maxpool_conv(x) + t_emb


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, time_dim, bilinear=True):
        super(Up, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_block = nn.Sequential(
            DoubleConv(in_channels + in_channels // 2, out_channels),
            DoubleConv(out_channels, out_channels),
        )

        self.time_emb = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))

    def forward(self, x1, x2, t):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)

        t_emb = self.time_emb(t)[:, :, None, None].repeat(
            1, 1, x.shape[-2], x.shape[-1]
        )

        return self.conv_block(x) + t_emb


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, out_channels, time_dim=128, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.time_dim = time_dim
        # self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128, time_dim=time_dim)
        self.self_attention_1 = SelfAttention(in_channels=128)
        self.down2 = Down(128, 256, time_dim=time_dim)
        self.self_attention_2 = SelfAttention(in_channels=256)
        self.down3 = Down(256, 512, time_dim=time_dim)
        self.self_attention_3 = SelfAttention(in_channels=512)
        self.up1 = Up(512, 256, time_dim=time_dim)
        self.self_attention_4 = SelfAttention(in_channels=256)
        self.up2 = Up(256, 128, time_dim=time_dim)
        self.self_attention_5 = SelfAttention(in_channels=128)
        self.up3 = Up(128, 64, time_dim=time_dim)
        self.self_attention_6 = SelfAttention(in_channels=64)
        self.outc = OutConv(64, out_channels)

        self.positional_encoding = PositionalEncoding(dim=self.time_dim)

    def forward(self, x, t):
        t = self.positional_encoding(t)

        x1 = self.inc(x)

        x2 = self.down1(x1, t)
        x2 = self.self_attention_1(x2)

        x3 = self.down2(x2, t)
        x3 = self.self_attention_2(x3)

        x4 = self.down3(x3, t)
        x4 = self.self_attention_3(x4)

        x = self.up1(x4, x3, t)
        x = self.self_attention_4(x)

        x = self.up2(x, x2, t)
        x = self.self_attention_5(x)

        x = self.up3(x, x1, t)
        x = self.self_attention_6(x)

        out = self.outc(x)
        return out


if __name__ == "__main__":
    # Example usage
    model = UNet(n_channels=3, out_channels=3)
    x = torch.randn(1, 3, 256, 256)  # Batch size of 1, 3 input channels, 256x256 image
    t = torch.tensor([64])
    y = model(x, t)
    print(y.shape)  # Should output torch.Size([1, 1, 256, 256])
