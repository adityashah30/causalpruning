import torch
import torch.nn as nn
import torch.nn.functional as F


def _calc_output_size(insize: int, kernel: int, stride: int) -> int:
    return int((insize - (kernel - 1) - 1) / stride) + 1


class LeNet(nn.Module):
    def __init__(self, num_classes: int, size_input: tuple[int, int, int], kernel: int):
        super().__init__()
        num_features, H, W = size_input
        self.conv1 = nn.Conv2d(num_features, 20, kernel, 1)
        H, W = _calc_output_size(H, kernel, 1), _calc_output_size(W, kernel, 1)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        H, W = _calc_output_size(H, 2, 2), _calc_output_size(W, 2, 2)
        self.conv2 = nn.Conv2d(20, 50, kernel, 1)
        H, W = _calc_output_size(H, kernel, 1), _calc_output_size(W, kernel, 1)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        H, W = _calc_output_size(H, 2, 2), _calc_output_size(W, 2, 2)
        self.size_flatten = 50 * H * W
        self.fc1 = nn.Linear(self.size_flatten, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = x.reshape(-1, self.size_flatten)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_lenet(dataset: str) -> nn.Module:
    dataset = dataset.lower()
    if dataset == "cifar10":
        return LeNet(num_classes=10, size_input=(3, 32, 32), kernel=5)
    elif dataset == "fashionmnist":
        return LeNet(num_classes=10, size_input=(1, 28, 28), kernel=5)
    elif dataset == "tinyimagenet":
        return LeNet(num_classes=200, size_input=(3, 64, 64), kernel=7)
    raise NotImplementedError(f"LeNet is not available for {dataset}")
