import torch
import torch.nn as nn
import torch.nn.functional as F


def _calc_output_size(
    insize: int, kernel: int, stride: int = 1, padding: int = 0
) -> int:
    return (insize + 2 * padding - (kernel - 1) - 1 + stride) // stride


class AlexNet_32(nn.Module):
    def __init__(self, num_classes: int, size_input: tuple[int, int, int]):
        super().__init__()
        num_channels, H, W = size_input
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=2, padding=1)
        H, W = _calc_output_size(H, 3, 2, 1), _calc_output_size(W, 3, 2, 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        H, W = _calc_output_size(H, 2, 2), _calc_output_size(W, 2, 2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=2, padding=1)
        H, W = _calc_output_size(H, 2, 1, 1), _calc_output_size(W, 2, 1, 1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        H, W = _calc_output_size(H, 2, 2), _calc_output_size(W, 2, 2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        H, W = _calc_output_size(H, 3, 1, 1), _calc_output_size(W, 3, 1, 1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        H, W = _calc_output_size(H, 3, 1, 1), _calc_output_size(W, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        H, W = _calc_output_size(H, 3, 1, 1), _calc_output_size(W, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        H, W = _calc_output_size(H, 2, 2), _calc_output_size(W, 2, 2)
        flattened_dims = 256 * H * W
        self.fc1 = nn.Linear(flattened_dims, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool3(x)
        x = torch.flatten(x, 1)
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AlexNet_ImageNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_alexnet(dataset: str) -> nn.Module:
    dataset = dataset.lower()
    if dataset == "cifar10":
        model = AlexNet_32(num_classes=10, size_input=(3, 32, 32))
        return model
    elif dataset == "fashionmnist":
        model = AlexNet_32(num_classes=10, size_input=(1, 28, 28))
        return model
    elif dataset == "tinyimagenet":
        model = AlexNet_ImageNet(num_classes=200)
        return model
    raise NotImplementedError(f"AlexNet is not available for {dataset}")
