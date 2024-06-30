from .cifar10 import get_cifar_10
from .mnist import get_mnist

import torch.utils.data as data


def get_dataset(dataset_name: str,
                model_name: str,
                root_dir: str,
                recompute: bool = False) -> tuple[data.Dataset, data.Dataset, int]:
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar10':
        train, test = get_cifar_10(model_name, root_dir, recompute)
        return (train, test, 10)
    elif dataset_name == 'mnist':
        train, test = get_mnist(model_name, root_dir, recompute)
        return (train, test, 10)
    raise NotImplementedError(f'{dataset_name} not available.')
