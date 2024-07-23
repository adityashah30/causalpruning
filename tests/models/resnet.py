import torch
import torch.nn as nn
from torchvision.models import resnet18


def initialize_model_weights(model: nn.Module) -> nn.Module:
    rand_genarator = torch.Generator().manual_seed(56)
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(
                m.weight,
                mode="fan_out",
                nonlinearity="relu",
                generator=rand_genarator,
            )
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    return model


def get_resnet18(dataset: str) -> nn.Module:
    dataset = dataset.lower()
    if dataset == 'cifar10':
        model = resnet18()
        model = initialize_model_weights(model)
        return model
    raise NotImplementedError(f'Resnet18 is not available for {dataset}')
