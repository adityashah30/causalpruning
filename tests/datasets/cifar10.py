import os
from typing import Optional, Callable

from stocaching import SharedCache
import torch
import torch.utils.data as data
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2


_DEFAULT_TRAIN_TRANSFORMS = [
    v2.RandomHorizontalFlip(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010))
]
_DEFAULT_TEST_TRANSFORMS = [
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010))
]


class CachedCIFAR10(CIFAR10):

    def __init__(self,
                 root_dir: str,
                 size: int,
                 train: bool,
                 transform: Optional[Callable],
                 download: bool,
                 cache_size_limit_gb: int):
        super().__init__(root_dir, train, transform, download=download)
        num_items = super().__len__()
        self.input_cache = SharedCache(
            size_limit_gib=cache_size_limit_gb,
            dataset_len=num_items,
            data_dims=(3, size, size),
            dtype=torch.float32)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.input_cache.get_slot(idx)
        if inputs is None:
            inputs, _ = super().__getitem__(idx)
            self.input_cache.set_slot(idx, inputs)
        labels = self.targets[idx]
        return (inputs, labels)


@torch.no_grad
def get_cifar_10(
        model_name: str,
        data_root_dir: str,
        cache_size_limit_gb: int) -> tuple[data.Dataset, data.Dataset]:
    model_name = model_name.lower()
    cifar_root_dir = os.path.join(data_root_dir, 'cifar10')
    if model_name in ['lenet', 'fullyconnected', 'resnet18']:
        train_transforms = v2.Compose(_DEFAULT_TRAIN_TRANSFORMS)
        test_transforms = v2.Compose(_DEFAULT_TEST_TRANSFORMS)
        train = CachedCIFAR10(cifar_root_dir,
                              size=32,
                              train=True,
                              transform=train_transforms,
                              download=True,
                              cache_size_limit_gb=cache_size_limit_gb)
        test = CachedCIFAR10(cifar_root_dir,
                             size=32,
                             train=False,
                             transform=test_transforms,
                             download=True,
                             cache_size_limit_gb=cache_size_limit_gb)
        return (train, test)
    elif model_name == 'alexnet':
        train_transforms = v2.Compose(
            [v2.Resize((227, 227))] + _DEFAULT_TRAIN_TRANSFORMS)
        test_transforms = v2.Compose(
            [v2.Resize((227, 227))] + _DEFAULT_TEST_TRANSFORMS)
        train = CachedCIFAR10(cifar_root_dir,
                              size=227,
                              train=True,
                              transform=train_transforms,
                              download=True,
                              cache_size_limit_gb=cache_size_limit_gb)
        test = CachedCIFAR10(cifar_root_dir,
                             size=227,
                             train=False,
                             transform=test_transforms,
                             download=True,
                             cache_size_limit_gb=cache_size_limit_gb)
        return (train, test)
    raise NotImplementedError(f'CIFAR10 not available for {model_name}')
