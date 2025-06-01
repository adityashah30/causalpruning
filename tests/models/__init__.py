from .alexnet import get_alexnet
from .lenet import get_lenet
from .mlpnet import get_mlpnet
from .resnet import (
    get_resnet18,
    get_resnet50,
    get_resnet50_untrained,
)
from .resnet_cifar import get_resnet20

import torch.nn as nn


def get_model(model_name: str, dataset_name: str) -> nn.Module:
    model_name, dataset_name = model_name.lower(), dataset_name.lower()
    if model_name == "lenet":
        return get_lenet(dataset_name)
    elif model_name == "alexnet":
        return get_alexnet(dataset_name)
    elif model_name == "mlpnet":
        return get_mlpnet(dataset_name)
    elif model_name == "resnet18":
        return get_resnet18(dataset_name)
    elif model_name == "resnet20":
        return get_resnet20(dataset_name)
    elif model_name == "resnet50":
        return get_resnet50(dataset_name)
    elif model_name == "resnet50_untrained":
        return get_resnet50_untrained(dataset_name)
    raise NotImplementedError(f"{model_name} is not implemented in the test suite")
