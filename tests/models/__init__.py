from typing import Optional

from .alexnet import get_alexnet
from .lenet import get_lenet
from .mlpnet import get_mlpnet
from .mobilenet import (
    get_mobilenet_trained,
    get_mobilenet_untrained,
)
from .resnet import (
    get_resnet18,
    get_resnet50_torch,
    get_resnet50_trained,
    get_resnet50_untrained,
)
from .resnet_cifar import get_resnet20

import torch.nn as nn


def get_model(model_name: str, dataset_name: str, checkpoint_dir: str) -> nn.Module:
    model_name, dataset_name = model_name.lower(), dataset_name.lower()
    if model_name == "lenet":
        return get_lenet(dataset_name)
    elif model_name == "alexnet":
        return get_alexnet(dataset_name)
    elif model_name == "mlpnet":
        return get_mlpnet(dataset_name)
    elif model_name == "mobilenet_untrained":
        return get_mobilenet_untrained(dataset_name)
    elif model_name == "mobilenet_trained":
        return get_mobilenet_trained(dataset_name, checkpoint_dir)
    elif model_name == "resnet18":
        return get_resnet18(dataset_name)
    elif model_name == "resnet20":
        return get_resnet20(dataset_name)
    elif model_name == "resnet50_torch":
        return get_resnet50_torch(dataset_name)
    elif model_name == "resnet50_untrained":
        return get_resnet50_untrained(dataset_name)
    elif model_name == "resnet50_trained":
        return get_resnet50_trained(dataset_name, checkpoint_dir)
    raise NotImplementedError(f"{model_name} is not implemented in the test suite")
