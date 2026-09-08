import math
import torch
import torch.nn as nn

class SinusoidalTimestepEmbed(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

        assert dim % 2 == 0

    def forward(self, timesteps):
        half_dim = self.dim // 2

        frequencies = torch.exp(
            -math.log(self.max_period) * torch.arange(start = 0, end=half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
        )

        args = timesteps[:, None].float() * frequencies[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return embedding

embed_layer = SinusoidalTimestepEmbed(dim=128)
t = torch.tensor([0, 100, 500, 999])
out = embed_layer(t)
print(out.shape)  # should be torch.Size([4, 128])