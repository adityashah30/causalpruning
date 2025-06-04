import os
import torch
import torch.nn as nn
from torchvision.models import (
    resnet18,
    resnet50,
    ResNet18_Weights,
    ResNet50_Weights,
)
from tqdm.auto import tqdm


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


def get_resnet50_torch(dataset: str) -> nn.Module:
    dataset = dataset.lower()
    if dataset in ["imagenet", "imagenet_memory"]:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        return model
    raise NotImplementedError(f"Resnet50 (torch) is not available for {dataset}")


def get_resnet50_trained(dataset: str, checkpoint_dir: str) -> nn.Module:
    dataset = dataset.lower()
    if dataset in ["imagenet", "imagenet_memory"]:
        model = resnet50()
        checkpoint_path = os.path.join(checkpoint_dir, "resnet50.pth")
        state_trained = torch.load(checkpoint_path, map_location=torch.device("cpu"))[
            "state_dict"
        ]
        new_state_trained = model.state_dict()
        for k in state_trained:
            key = k[7:]
            if key in new_state_trained:
                new_state_trained[key] = state_trained[k].view(
                    new_state_trained[key].size()
                )
            else:
                print("Missing key", key)
        model.load_state_dict(new_state_trained, strict=False)
        tqdm.write(f"Loaded Resnet50 weights from {checkpoint_path}")
        return model
    raise NotImplementedError(f"Resnet50 (trained) is not available for {dataset}")


def get_resnet50_untrained(dataset: str) -> nn.Module:
    dataset = dataset.lower()
    if dataset in ["imagenet", "imagenet_memory"]:
        model = resnet50()
        return model
    raise NotImplementedError(f"Resnet50 (untrained) is not available for {dataset}")
