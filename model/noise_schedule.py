import torch

class NoiseSchedule():
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.timesteps = timesteps

        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)

        self.alphas = 1.0 - self.betas

        self.alphas.cumprod = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas.cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alphas.cumprod)

    def add_noise(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)

        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise