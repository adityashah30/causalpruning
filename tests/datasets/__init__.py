from .cifar10 import TransformedCIFAR10

import torch
import torch.utils.data as data
from torchvision.transforms import v2


def get_dataset(dataset_name: str,
                model_name: str,
                root_dir: str,
                recompute: bool = False) -> tuple[data.Dataset, data.Dataset]:
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar10':
        if model_name == 'lenet':
            train = TransformedCIFAR10(
                root_dir, size=32, train=True, recompute=recompute,
                transforms=[
                    v2.RandomHorizontalFlip(),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(
                        mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2023, 0.1994, 0.2010))
                ])
            test = TransformedCIFAR10(
                root_dir, size=32, train=False, recompute=recompute,
                transforms=[
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(
                        mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2023, 0.1994, 0.2010))
                ])
            return (train, test)
        elif model_name == 'alexnet':
            train = TransformedCIFAR10(
                root_dir, size=64, train=True, recompute=recompute,
                transforms=[
                    v2.Resize((70, 70)),
                    v2.RandomCrop((64, 64)),
                    v2.RandomHorizontalFlip(),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(
                        mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2023, 0.1994, 0.2010))
                ])
            test = TransformedCIFAR10(
                root_dir, size=64, train=False, recompute=recompute,
                transforms=[
                    v2.Resize((64, 64)),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(
                        mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2023, 0.1994, 0.2010))
                ])
            return (train, test)
        raise NotImplementedError(
            f'{dataset_name} not available for {model_name}')
    raise NotImplementedError(
        f'{dataset_name} not available for {model_name}')
