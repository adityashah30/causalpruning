import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1,
            bias=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(
                in_planes, self.expansion * planes, kernel_size=1,
                stride=stride, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
            self, block, num_blocks, num_classes, kernel: int, stride: int,
            padding: int):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=kernel,
                               stride=stride, padding=padding, bias=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(
        n_classes: int, kernel: int, stride: int, padding: int) -> nn.Module:
    return ResNet(BasicBlock, [2, 2, 2, 2], n_classes, kernel, stride, padding)


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
        model = ResNet18(n_classes=10, kernel=3, stride=1, padding=1)
        model = initialize_model_weights(model)
        return model
    raise NotImplementedError(f'Resnet18 is not available for {dataset}')


def get_resnet50(dataset: str) -> nn.Module:
    dataset = dataset.lower()
    if dataset == 'imagenet':
        model = resnet50(ResNet50_Weights.IMAGENET1K_V2)
        return model
    raise NotImplementedError(f'Resnet50 is not available for {dataset}')
