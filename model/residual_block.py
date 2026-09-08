import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)

        self.time_proj = nn.Linear(time_embed_dim, out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.activation = nn.SiLU()

        self.skip = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_embed):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)

        time_bias = self.time_proj(t_embed)[:, :, None, None]
        h = h + time_bias

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)

        return h + self.skip(x)