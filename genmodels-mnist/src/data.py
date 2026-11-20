import torch
from torch.utils.data import DataLoader, TensorDataset


def get_loader(batch_size=128, fake=False, root="data"):
    """MNIST normalisé dans [-1, 1]. `fake=True` : données synthétiques (test rapide du code)."""
    if fake:
        x = torch.rand(512, 1, 28, 28) * 2 - 1
        return DataLoader(TensorDataset(x, torch.zeros(512)), batch_size=batch_size, shuffle=True)
    from torchvision import datasets, transforms
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds = datasets.MNIST(root, train=True, download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
