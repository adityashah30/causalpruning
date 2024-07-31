import torch
import torch.nn as nn
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
)


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
            bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1,
            bias=True)
        self.downsample = None
        if downsample:
            self.downsample = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride,
                bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x.clone()
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.downsample:
            identity = self.downsample(identity)
        x += identity
        return self.relu(x)


class Resnet18(nn.Module):
    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=(3, 3),
            bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2, downsample=True),
            BasicBlock(128, 128),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=2, downsample=True),
            BasicBlock(256, 256),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride=2, downsample=True),
            BasicBlock(512, 512),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)  # [bs, 512, 1, 1]
        x = torch.squeeze(x)  # reshape to [bs, 512]
        return self.fc(x)


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
        model = Resnet18(n_classes=10)
        model = initialize_model_weights(model)
        return model
    raise NotImplementedError(f'Resnet18 is not available for {dataset}')


def get_resnet50(dataset: str) -> nn.Module:
    dataset = dataset.lower()
    if dataset == 'imagenet':
        model = resnet50(ResNet50_Weights.IMAGENET1K_V2)
        return model
    raise NotImplementedError(f'Resnet50 is not available for {dataset}')
