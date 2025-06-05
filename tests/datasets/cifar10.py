import os

import torch
import torch.utils.data as data
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2


@torch.no_grad
def get_cifar_10(
    model_name: str, data_root_dir: str
) -> tuple[data.Dataset, data.Dataset]:
    model_name = model_name.lower()
    cifar_root_dir = os.path.join(data_root_dir, "cifar10")
    if model_name in ["alexnet", "lenet", "mlpnet", "resnet18", "resnet20"]:
        train_transforms = v2.Compose(
            [
                v2.RandomCrop(32),
                v2.RandomHorizontalFlip(p=0.5),
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
                v2.RandomErasing(0.1),
                v2.ToPureTensor(),
            ]
        )
        test_transforms = v2.Compose(
            [
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
                v2.ToPureTensor(),
            ]
        )
        train = CIFAR10(
            cifar_root_dir, train=True, download=True, transform=train_transforms
        )
        test = CIFAR10(
            cifar_root_dir, train=False, download=True, transform=test_transforms
        )
        return (train, test)
    raise NotImplementedError(f"CIFAR10 not available for {model_name}")
