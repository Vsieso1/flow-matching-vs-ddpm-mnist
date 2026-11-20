"""Flow matching (trajectoires linéaires / rectified flow).

Chemin : x_t = (1 - t) * x0 + t * x1, avec x0 ~ N(0, I) (bruit) et x1 ~ données.
Champ de vitesse cible : u_t = x1 - x0.
Le réseau v_theta(x_t, t) est entraîné par régression : || v_theta - (x1 - x0) ||^2.
Génération : intégrer dx/dt = v_theta(x, t) de t=0 (bruit) à t=1 (image).
"""
import torch
import torch.nn.functional as F


def fm_loss(model, x1):
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.size(0), device=x1.device)
    t_ = t[:, None, None, None]
    xt = (1 - t_) * x0 + t_ * x1
    target = x1 - x0
    pred = model(xt, t)
    return F.mse_loss(pred, target)


@torch.no_grad()
def fm_sample(model, shape, steps=50, device="cpu"):
    """Intégration d'Euler explicite avec `steps` pas."""
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((shape[0],), i * dt, device=device)
        x = x + model(x, t) * dt
    return x.clamp(-1, 1)
