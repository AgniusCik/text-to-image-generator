import torch
import torch.nn as nn
from sinusoidal_timestep_embed import SinusoidalTimestepEmbed
from residual_block import ResidualBlock
from cross_attention import CrossAttention

class UNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, time_embed_dim=256, text_dim=512):
        super().__init__()

        self.time_embed = SinusoidalTimestepEmbed(dim=time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        self.down1 = ResidualBlock(in_channels, base_channels, time_embed_dim)
        self.pool1 = nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)

        self.down2 = ResidualBlock(base_channels, base_channels * 2, time_embed_dim)
        self.cross_attn2 = CrossAttention(base_channels * 2, text_dim)  # NEW
        self.pool2 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=4, stride=2, padding=1)

        self.bottleneck = ResidualBlock(base_channels * 2, base_channels * 4, time_embed_dim)
        self.cross_attn_bottleneck = CrossAttention(base_channels * 4, text_dim)  # NEW

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.up_block2 = ResidualBlock(base_channels * 4, base_channels * 2, time_embed_dim)
        self.cross_attn_up2 = CrossAttention(base_channels * 2, text_dim)  # NEW

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.up_block1 = ResidualBlock(base_channels * 2, base_channels, time_embed_dim)

        self.out = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x, t, text_embed):
        t_embed = self.time_embed(t)
        t_embed = self.time_mlp(t_embed)

        d1 = self.down1(x, t_embed)
        p1 = self.pool1(d1)

        d2 = self.down2(p1, t_embed)
        d2 = self.cross_attn2(d2, text_embed)  # NEW
        p2 = self.pool2(d2)

        b = self.bottleneck(p2, t_embed)
        b = self.cross_attn_bottleneck(b, text_embed)  # NEW

        u2 = self.up2(b)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up_block2(u2, t_embed)
        u2 = self.cross_attn_up2(u2, text_embed)  # NEW

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.up_block1(u1, t_embed)

        return self.out(u1)