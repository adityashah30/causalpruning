import os
from typing import Optional, Callable

from stocaching import SharedCache
import torch
import torch.utils.data as data
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2


_DEFAULT_TRAIN_TRANSFORMS = [
    v2.RandomHorizontalFlip(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225))
]
_DEFAULT_TEST_TRANSFORMS = [
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225))
]


class CachedTinyImageNet(ImageFolder):

    def __init__(self,
                 root_dir: str,
                 size: int,
                 train: bool,
                 transform: Optional[Callable],
                 cache_size_limit_gb: int):
        split = 'train' if train else 'test'
        tinyimagenet_root_dir = os.path.join(root_dir, split)
        super().__init__(tinyimagenet_root_dir, transform=transform)
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
def get_tiny_imagenet(
        model_name: str,
        data_root_dir: str,
        cache_size_limit_gb: int) -> tuple[data.Dataset, data.Dataset]:
    model_name = model_name.lower()
    tinyimagenet_root_dir = os.path.join(data_root_dir, 'tinyimagenet200')
    if model_name in ['lenet', 'resnet18']:
        train_transforms = v2.Compose(_DEFAULT_TRAIN_TRANSFORMS)
        test_transforms = v2.Compose(_DEFAULT_TEST_TRANSFORMS)
        train = CachedTinyImageNet(tinyimagenet_root_dir,
                                   size=64,
                                   train=True,
                                   transform=train_transforms,
                                   cache_size_limit_gb=cache_size_limit_gb)
        test = CachedTinyImageNet(tinyimagenet_root_dir,
                                  size=64,
                                  train=False,
                                  transform=test_transforms,
                                  cache_size_limit_gb=cache_size_limit_gb)
        return (train, test)
    elif model_name == 'alexnet':
        train_transforms = v2.Compose(
            [v2.Resize((227, 227))] + _DEFAULT_TRAIN_TRANSFORMS)
        test_transforms = v2.Compose(
            [v2.Resize((227, 227))] + _DEFAULT_TEST_TRANSFORMS)
        train = CachedTinyImageNet(tinyimagenet_root_dir,
                                   size=227,
                                   train=True,
                                   transform=train_transforms,
                                   cache_size_limit_gb=cache_size_limit_gb)
        test = CachedTinyImageNet(tinyimagenet_root_dir,
                                  size=227,
                                  train=False,
                                  transform=test_transforms,
                                  cache_size_limit_gb=cache_size_limit_gb)
        return (train, test)
    raise NotImplementedError(f'TinyImageNet not available for {model_name}')
