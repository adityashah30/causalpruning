import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet18,
    resnet50,
    ResNet18_Weights,
    ResNet50_Weights,
)


def get_resnet18(dataset: str) -> nn.Module:
    dataset = dataset.lower()
    if dataset == "cifar10":
        model = resnet18(weights=None, num_classes=10)
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        model.maxpool = nn.Identity()
        return model
    elif dataset == "tinyimagenet":
        model = resnet18(weights=None, num_classes=200)
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        model.maxpool = nn.Identity()
        return model
    elif dataset == "imagenet":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1k_V1)
    raise NotImplementedError(f"Resnet18 is not available for {dataset}")


def get_resnet20(dataset: str) -> nn.Module:
    dataset = dataset.lower()
    raise NotImplementedError(f"Resnet20 is not available for {dataset}")


def get_resnet50(dataset: str) -> nn.Module:
    dataset = dataset.lower()
    if dataset == "imagenet":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        return model
    raise NotImplementedError(f"Resnet50 is not available for {dataset}")


def get_resnet50_untrained(dataset: str) -> nn.Module:
    dataset = dataset.lower()
    if dataset == "imagenet":
        model = resnet50()
        model = initialize_model_weights(model)
        return model
    raise NotImplementedError(
        f"Resnet50 (untrained) is not available for {dataset}")
