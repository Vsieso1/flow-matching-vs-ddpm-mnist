"""Petit U-Net conditionné par le temps, pour des images 1x28x28."""
import math
import torch
import torch.nn as nn


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Embedding sinusoïdal du temps (t de forme [B], valeurs réelles)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class Block(nn.Module):
    def __init__(self, c_in, c_out, t_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, c_out)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.t_proj = nn.Linear(t_dim, c_out)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.t_proj(t_emb)[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, base=32, t_dim=128):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.inp = nn.Conv2d(1, base, 3, padding=1)
        self.d1 = Block(base, base, t_dim)            # 28x28
        self.d2 = Block(base, base * 2, t_dim)        # 14x14
        self.mid = Block(base * 2, base * 4, t_dim)   # 7x7
        self.u2 = Block(base * 4 + base * 2, base * 2, t_dim)  # 14x14
        self.u1 = Block(base * 2 + base, base, t_dim)           # 28x28
        self.out = nn.Conv2d(base, 1, 1)
        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, t):
        te = self.t_mlp(timestep_embedding(t, self.t_dim))
        x0 = self.inp(x)
        h1 = self.d1(x0, te)                    # [B, b, 28, 28]
        h2 = self.d2(self.pool(h1), te)         # [B, 2b, 14, 14]
        m = self.mid(self.pool(h2), te)         # [B, 4b, 7, 7]
        u = self.u2(torch.cat([self.up(m), h2], 1), te)
        u = self.u1(torch.cat([self.up(u), h1], 1), te)
        return self.out(u)
