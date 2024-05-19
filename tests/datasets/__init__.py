from .cifar10 import get_cifar10

import torch.utils.data as data


def get_dataset(dataset_name: str, root_dir: str, download: bool = True) -> tuple[data.Dataset, data.Dataset]:
    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar10':
        return get_cifar10(root_dir, download)
    raise NotImplementedError(f"{dataset_name} not implemented yet.")
