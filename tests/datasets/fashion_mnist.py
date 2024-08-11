import os
from typing import Optional, Callable

from stocaching import SharedCache
import torch
import torch.utils.data as data
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2


_DEFAULT_TRANSFORMS = [
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.2860,), std=(0.3530,))
]


class CachedFashionMNIST(FashionMNIST):

    def __init__(self,
                 root_dir: str,
                 size: int,
                 train: bool,
                 transform: Optional[Callable],
                 download: bool,
                 cache_size_limit_gb):
        super().__init__(root_dir, train, transform, download=download)
        num_items = super().__len__()
        self.input_cache = SharedCache(
            size_limit_gib=cache_size_limit_gb,
            dataset_len=num_items,
            data_dims=(1, size, size),
            dtype=torch.float32)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.input_cache.get_slot(idx)
        if inputs is None:
            inputs, _ = super().__getitem__(idx)
            self.input_cache.set_slot(idx, inputs)
        labels = self.targets[idx]
        return (inputs, labels)

@torch.no_grad
def get_fashion_mnist(
        model_name: str,
        data_root_dir: str,
        cache_size_limit_gb: int) -> tuple[data.Dataset, data.Dataset]:
    model_name = model_name.lower()
    fashionmnist_root_dir = os.path.join(data_root_dir, 'fashionmnist')
    if model_name == 'lenet':
        transform = v2.Compose(_DEFAULT_TRANSFORMS)
        train = CachedFashionMNIST(fashionmnist_root_dir,
                                   size=28,
                                   train=True,
                                   transform=transform,
                                   download=True,
                                   cache_size_limit_gb=cache_size_limit_gb)
        test = CachedFashionMNIST(fashionmnist_root_dir,
                                  size=28,
                                  train=False,
                                  transform=transform,
                                  download=True,
                                  cache_size_limit_gb=cache_size_limit_gb)
        return (train, test)
    elif model_name == 'resnet18':
        transform = v2.Compose([v2.Resize((32, 32))] + _DEFAULT_TRANSFORMS)
        train = CachedFashionMNIST(fashionmnist_root_dir,
                                   size=32,
                                   train=True,
                                   transform=transform,
                                   download=True,
                                   cache_size_limit_gb=cache_size_limit_gb)
        test = CachedFashionMNIST(fashionmnist_root_dir,
                                  size=32,
                                  train=False,
                                  transform=transform,
                                  download=True,
                                  cache_size_limit_gb=cache_size_limit_gb)
        return (train, test)
    elif model_name == 'alexnet':
        transform = v2.Compose([v2.Resize((227, 227))] + _DEFAULT_TRANSFORMS)
        train = CachedFashionMNIST(fashionmnist_root_dir,
                                   size=227,
                                   train=True,
                                   transform=transform,
                                   download=True,
                                   cache_size_limit_gb=cache_size_limit_gb)
        test = CachedFashionMNIST(fashionmnist_root_dir,
                                  size=227,
                                  train=False,
                                  transform=transform,
                                  download=True,
                                  cache_size_limit_gb=cache_size_limit_gb)
        return (train, test)
    raise NotImplementedError(f'CIFAR10 not available for {model_name}')
