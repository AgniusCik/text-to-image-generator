import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, image_channels, text_dim, num_heads=4):
        super().__init__()

        self.norm = nn.GroupNorm(8, image_channels)
        self.to_q = nn.Conv2d(image_channels, image_channels, kernel_size=1)
        self.to_k = nn.Linear(text_dim, image_channels)
        self.to_v = nn.Linear(text_dim, image_channels)
        self.attn = nn.MultiheadAttention(image_channels, num_heads, batch_first=True)
        self.to_out = nn.Conv2d(image_channels, image_channels, kernel_size=1)

    def forward(self, x, text_embed):
        B, C, H, W = x.shape
        residual = x

        x = self.norm(x)
        q = self.to_q(x).view(B, C, H * W).permute(0, 2, 1)
        k = self.to_k(text_embed)
        v = self.to_v(text_embed)

        attn_out, _ = self.attn(q, k, v)
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)
        attn_out = self.to_out(attn_out)

        return residual + attn_out