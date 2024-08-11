from .cifar10 import get_cifar_10
from .fashion_mnist import get_fashion_mnist
from .imagenet import get_imagenet

import torch
import torch.utils.data as data


@torch.no_grad
def get_dataset(dataset_name: str,
                model_name: str,
                data_root_dir: str,
                cache_size_limit_gb: int) -> tuple[
                    data.Dataset, data.Dataset, int]:
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar10':
        train, test = get_cifar_10(
            model_name, data_root_dir, cache_size_limit_gb)
        return (train, test, 10)
    elif dataset_name == 'fashionmnist':
        train, test = get_fashion_mnist(
            model_name, data_root_dir, cache_size_limit_gb)
        return (train, test, 10)
    elif dataset_name == 'imagenet':
        train, test = get_imagenet(
            model_name, data_root_dir)
        return (train, test, 1000)
    raise NotImplementedError(f'{dataset_name} not available.')
