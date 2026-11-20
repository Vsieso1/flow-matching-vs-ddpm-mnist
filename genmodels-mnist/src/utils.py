import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_grid(x, path, nrow=8):
    x = (x.detach().cpu() + 1) / 2
    n = x.size(0)
    ncol = (n + nrow - 1) // nrow
    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow, ncol))
    for ax, img in zip(axes.flatten(), list(x) + [None] * (nrow * ncol - n)):
        ax.axis("off")
        if img is not None:
            ax.imshow(img[0], cmap="gray", vmin=0, vmax=1)
    plt.tight_layout(pad=0.1)
    plt.savefig(path, dpi=100)
    plt.close(fig)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
