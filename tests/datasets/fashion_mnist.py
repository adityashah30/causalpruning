import os

import torch
import torch.utils.data as data
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2


_DEFAULT_TRANSFORMS = [
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.2860,), std=(0.3530,)),
]


@torch.no_grad
def get_fashion_mnist(
    model_name: str, data_root_dir: str
) -> tuple[data.Dataset, data.Dataset]:
    model_name = model_name.lower()
    fashionmnist_root_dir = os.path.join(data_root_dir, "fashionmnist")
    if model_name == "lenet":
        transform = v2.Compose(_DEFAULT_TRANSFORMS)
        train = FashionMNIST(
            fashionmnist_root_dir, train=True, download=True, transform=transform
        )
        test = FashionMNIST(
            fashionmnist_root_dir, train=False, download=True, transform=transform
        )
        return (train, test)
    elif model_name in ["alexnet", "resnet18"]:
        transform = v2.Compose([v2.Resize((32, 32))] + _DEFAULT_TRANSFORMS)
        train = FashionMNIST(
            fashionmnist_root_dir, train=True, download=True, transform=transform
        )
        test = FashionMNIST(
            fashionmnist_root_dir, train=False, download=True, transform=transform
        )
        return (train, test)
    raise NotImplementedError(f"FashionMNIST not available for {model_name}")
