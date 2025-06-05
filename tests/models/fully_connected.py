import torch
import torch.nn as nn


class FullyConnected(nn.Module):
    def __init__(self, num_classes: int, size_input: tuple[int, int, int]):
        super().__init__()
        num_features, H, W = size_input
        flattened_dims = num_features * H * W
        self.model = nn.Sequential(
            nn.Linear(flattened_dims, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x) -> torch.Tensor:
        num_dims = len(x.shape)
        if num_dims == 3:
            start_dim = 0
        elif num_dims == 4:
            start_dim = 1
        x = torch.flatten(x, start_dim=start_dim, end_dim=-1)
        return self.model(x)


def get_fully_connected(dataset: str) -> nn.Module:
    dataset = dataset.lower()
    if dataset == "cifar10":
        return FullyConnected(num_classes=10, size_input=(3, 32, 32))
    elif dataset == "fashionmnist":
        return FullyConnected(num_classes=10, size_input=(1, 28, 28))
    raise NotImplementedError(f"FullyConnected is not available for {dataset}")
