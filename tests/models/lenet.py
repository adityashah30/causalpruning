import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):

    def __init__(self, num_classes: int,
                 size_input: tuple[int, int, int]):
        super().__init__()
        num_features, H, W = size_input
        self.conv1 = nn.Conv2d(num_features, 20, 5, 1)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.size_flatten = int(
            50 * (((H - 4) / 2) - 4) / 2 * (((W - 4) / 2) - 4) / 2)
        self.fc1 = nn.Linear(self.size_flatten, 500)
        self.fc2 = nn.Linear(500, num_classes)

        rand_genarator = torch.Generator().manual_seed(56)
        # initialize weights using kaiming normal
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                    generator=rand_genarator,
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.reshape(-1, self.size_flatten)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_lenet(dataset: str) -> nn.Module:
    dataset = dataset.lower()
    if dataset == 'cifar10':
        return LeNet(num_classes=10, size_input=(3, 32, 32))
    elif dataset == 'fashionmnist':
        return LeNet(num_classes=10, size_input=(1, 28, 28))
    elif dataset == 'tinyimagenet':
        return LeNet(num_classes=200, size_input=(3, 64, 64))
    raise NotImplementedError(f'LeNet is not available for {dataset}')
