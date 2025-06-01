import os

import torch
import torch.utils.data as data
from torchvision.datasets import MNIST
from torchvision.transforms import v2


_DEFAULT_TRANSFORMS = [
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.1307,), std=(0.3081,)),
]


@torch.no_grad()
def get_mnist(model_name: str, data_root_dir: str) -> tuple[data.Dataset, data.Dataset]:
    model_name = model_name.lower()
    mnist_root_dir = os.path.join(data_root_dir, "mnist")
    if model_name == "mlpnet":
        transform = v2.Compose(_DEFAULT_TRANSFORMS)
        train = MNIST(mnist_root_dir, train=True, download=True, transform=transform)
        test = MNIST(mnist_root_dir, train=False, download=True, transform=transform)
        return (train, test)
    raise NotImplementedError(f"MNIST not available for {model_name}")
