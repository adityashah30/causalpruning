from .cifar10 import get_cifar_10

import torch.utils.data as data


def get_dataset(dataset_name: str,
                model_name: str,
                root_dir: str,
                recompute: bool = False) -> tuple[data.Dataset, data.Dataset]:
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar10':
        return get_cifar_10(model_name, root_dir, recompute)
    raise NotImplementedError(f'{dataset_name} not available.')
