"""Génération depuis un checkpoint. Compare le nombre de pas pour flow matching.
python sample.py --method flow --steps 5 10 50
"""
import argparse, os
import torch
from src.unet import UNet
from src.flow_matching import fm_sample
from src.ddpm import DDPM
from src.utils import save_grid, get_device

p = argparse.ArgumentParser()
p.add_argument("--method", choices=["flow", "ddpm"], default="flow")
p.add_argument("--steps", type=int, nargs="+", default=[50])
p.add_argument("--n", type=int, default=32)
p.add_argument("--out", default="outputs")
a = p.parse_args()

dev = get_device()
model = UNet().to(dev)
model.load_state_dict(torch.load(os.path.join(a.out, a.method, "ema.pt"), map_location=dev))
model.eval()
shape = (a.n, 1, 28, 28)
if a.method == "flow":
    for s in a.steps:
        torch.manual_seed(0)
        save_grid(fm_sample(model, shape, s, dev), os.path.join(a.out, a.method, f"gen_{s}steps.png"))
        print("ok", s, "pas")
else:
    save_grid(DDPM(device=dev).sample(model, shape), os.path.join(a.out, a.method, "gen_ddpm.png"))
