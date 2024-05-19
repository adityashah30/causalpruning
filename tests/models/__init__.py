from .lenet import LeNet

import torch.nn as nn

def get_model(model_name: str, dataset_name: str) -> nn.Module:
    model_name, dataset_name = model_name.lower(), dataset_name.lower()
    if model_name == 'lenet' and dataset_name == 'cifar10':
        return LeNet(num_classes=10, size_input=(3, 32, 32))
    raise NotImplementedError(
        f"{model_name} is not implemented in the test suite")
