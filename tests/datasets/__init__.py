from .cifar10 import TransformedCIFAR10

import torch.utils.data as data


def get_dataset(dataset_name: str, root_dir: str, recompute: bool = False) -> tuple[data.Dataset, data.Dataset]:
    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar10':
        train = TransformedCIFAR10(root_dir, train=True, recompute=recompute)
        test = TransformedCIFAR10(root_dir, train=False, recompute=recompute)
        return (train, test)
    raise NotImplementedError(f"{dataset_name} not implemented yet.")
