"""Entraînement : python train.py --method flow|ddpm --epochs 20"""
import argparse, os, copy
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.unet import UNet
from src.flow_matching import fm_loss, fm_sample
from src.ddpm import DDPM
from src.data import get_loader
from src.utils import save_grid, get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["flow", "ddpm"], default="flow")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--steps", type=int, default=50, help="pas d'échantillonnage (flow)")
    p.add_argument("--fake", action="store_true", help="données synthétiques (test du code)")
    p.add_argument("--out", default="outputs")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    dev = get_device()
    out = os.path.join(a.out, a.method)
    os.makedirs(out, exist_ok=True)
    print(f"device={dev} method={a.method}")

    loader = get_loader(a.batch_size, fake=a.fake)
    model = UNet().to(dev)
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    ddpm = DDPM(device=dev) if a.method == "ddpm" else None
    print(f"paramètres : {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    losses = []
    for ep in range(a.epochs):
        run = 0.0
        for x, _ in tqdm(loader, desc=f"epoch {ep+1}/{a.epochs}"):
            x = x.to(dev)
            loss = fm_loss(model, x) if a.method == "flow" else ddpm.loss(model, x)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():  # moyenne exponentielle des poids (EMA)
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(0.999).add_(pm, alpha=0.001)
            run += loss.item()
        losses.append(run / len(loader))
        print(f"epoch {ep+1}: loss={losses[-1]:.4f}")

        if (ep + 1) % 5 == 0 or ep == a.epochs - 1:
            shape = (16, 1, 28, 28)
            xs = fm_sample(ema, shape, a.steps, dev) if a.method == "flow" else ddpm.sample(ema, shape)
            save_grid(xs, os.path.join(out, f"samples_ep{ep+1}.png"), nrow=8)
            torch.save(ema.state_dict(), os.path.join(out, "ema.pt"))

    plt.figure(); plt.plot(losses); plt.xlabel("epoch"); plt.ylabel("loss"); plt.title(a.method)
    plt.savefig(os.path.join(out, "loss.png"), dpi=100)


if __name__ == "__main__":
    main()
