"""DDPM (Ho et al., 2020) : prédiction du bruit, schedule linéaire."""
import torch
import torch.nn.functional as F


class DDPM:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.abar = torch.cumprod(self.alphas, dim=0)

    def loss(self, model, x0):
        b = x0.size(0)
        t = torch.randint(0, self.T, (b,), device=x0.device)
        eps = torch.randn_like(x0)
        ab = self.abar[t][:, None, None, None]
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        # le modèle reçoit t normalisé dans [0, 1]
        pred = model(xt, t.float() / self.T)
        return F.mse_loss(pred, eps)

    @torch.no_grad()
    def sample(self, model, shape):
        x = torch.randn(shape, device=self.device)
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=self.device)
            eps = model(x, t.float() / self.T)
            a, ab, b = self.alphas[i], self.abar[i], self.betas[i]
            mean = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
            x = mean + b.sqrt() * torch.randn_like(x) if i > 0 else mean
        return x.clamp(-1, 1)
