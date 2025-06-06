import os
from typing import Callable, Optional

import torch
import torch.utils.data as data
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2


class TinyImageNet(ImageFolder):
    def __init__(
        self,
        root_dir: str,
        train: bool,
        transform: Optional[Callable],
    ):
        split = "train" if train else "test"
        tinyimagenet_root_dir = os.path.join(root_dir, split)
        super().__init__(tinyimagenet_root_dir, transform=transform)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, _ = super().__getitem__(idx)
        labels = self.targets[idx]
        return (inputs, labels)


@torch.no_grad
def get_tiny_imagenet(
    model_name: str, data_root_dir: str
) -> tuple[data.Dataset, data.Dataset]:
    model_name = model_name.lower()
    tinyimagenet_root_dir = os.path.join(data_root_dir, "tinyimagenet200")
    if model_name in ["lenet", "resnet18"]:
        train_transforms = v2.Compose(
            [
                v2.RandomResizedCrop(64),
                v2.RandomHorizontalFlip(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
            ]
        )
        test_transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
            ]
        )
        train = TinyImageNet(
            tinyimagenet_root_dir,
            train=True,
            transform=train_transforms,
        )
        test = TinyImageNet(
            tinyimagenet_root_dir,
            train=False,
            transform=test_transforms,
        )
        return (train, test)
    elif model_name == "alexnet":
        train_transforms = v2.Compose(
            [
                v2.RandomResizedCrop(224),
                v2.RandomHorizontalFlip(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
            ]
        )
        test_transforms = v2.Compose(
            [
                v2.Resize((224, 224)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
            ]
        )
        train = TinyImageNet(
            tinyimagenet_root_dir,
            train=True,
            transform=train_transforms,
        )
        test = TinyImageNet(
            tinyimagenet_root_dir,
            train=False,
            transform=test_transforms,
        )
        return (train, test)
    raise NotImplementedError(f"TinyImageNet not available for {model_name}")
