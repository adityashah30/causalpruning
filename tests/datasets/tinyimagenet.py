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
        mean=(0.4802, 0.4481, 0.3975),
        std=(0.2764, 0.2689, 0.2816))
]
_DEFAULT_TEST_TRANSFORMS = [
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=(0.4802, 0.4481, 0.3975),
        std=(0.2764, 0.2689, 0.2816))
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
    if model_name == 'lenet':
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
            [v2.Resize((224, 224))] + _DEFAULT_TRAIN_TRANSFORMS)
        test_transforms = v2.Compose(
            [v2.Resize((224, 224))] + _DEFAULT_TEST_TRANSFORMS)
        train = CachedTinyImageNet(tinyimagenet_root_dir,
                                   size=224,
                                   train=True,
                                   transform=train_transforms,
                                   cache_size_limit_gb=cache_size_limit_gb)
        test = CachedTinyImageNet(tinyimagenet_root_dir,
                                  size=224,
                                  train=False,
                                  transform=test_transforms,
                                  cache_size_limit_gb=cache_size_limit_gb)
        return (train, test)
    elif model_name == 'resnet18':
        train_transforms = v2.Compose(
            [v2.Resize((32, 32))] + _DEFAULT_TRAIN_TRANSFORMS)
        test_transforms = v2.Compose(
            [v2.Resize((32, 32))] + _DEFAULT_TEST_TRANSFORMS)
        train = CachedTinyImageNet(tinyimagenet_root_dir,
                                   size=32,
                                   train=True,
                                   transform=train_transforms,
                                   cache_size_limit_gb=cache_size_limit_gb)
        test = CachedTinyImageNet(tinyimagenet_root_dir,
                                  size=32,
                                  train=False,
                                  transform=test_transforms,
                                  cache_size_limit_gb=cache_size_limit_gb)
        return (train, test)
    raise NotImplementedError(f'TinyImageNet not available for {model_name}')
