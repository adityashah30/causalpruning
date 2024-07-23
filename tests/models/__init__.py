from .alexnet import get_alexnet
from .lenet import get_lenet
from .resnet import get_resnet18

import torch.nn as nn


def get_model(model_name: str, dataset_name: str) -> nn.Module:
    model_name, dataset_name = model_name.lower(), dataset_name.lower()
    if model_name == 'lenet':
        return get_lenet(dataset_name)
    elif model_name == 'alexnet':
        return get_alexnet(dataset_name)
    elif model_name == 'resnet18':
        return get_resnet18(dataset_name)
    raise NotImplementedError(
        f'{model_name} is not implemented in the test suite')
