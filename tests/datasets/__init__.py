from .cifar10 import TransformedCIFAR10

import torch.utils.data as data


def get_dataset(dataset_name: str, 
                model_name: str, 
                root_dir: str, 
                recompute: bool = False) -> tuple[data.Dataset, data.Dataset]:
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar10':
        if model_name == 'lenet':
            train = TransformedCIFAR10(
                root_dir, size=32, train=True, recompute=recompute)
            test = TransformedCIFAR10(
                root_dir, size=32, train=False, recompute=recompute)
            return (train, test)
        elif model_name == 'alexnet':
            train = TransformedCIFAR10(
                root_dir, size=64, train=True, recompute=recompute)
            test = TransformedCIFAR10(
                root_dir, size=64, train=False, recompute=recompute)
            return (train, test)
        raise NotImplementedError(
            f'{dataset_name} not available for {model_name}')
    raise NotImplementedError(
            f'{dataset_name} not available for {model_name}')
