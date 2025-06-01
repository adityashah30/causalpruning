import torch.nn as nn
import torch.nn.functional as F


class MlpNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        nh1: int,
        nh2: int,
        num_classes: int,
        bias: bool = False,
        enable_dropout: bool = False,
        log_softmax: bool = True,
    ):
        super().__init__()
        self.log_softmax = log_softmax
        self.fc1 = nn.Linear(input_size, nh1, bias=bias)
        self.fc2 = nn.Linear(nh1, nh2, bias=bias)
        self.fc3 = nn.Linear(nh2, num_classes, bias=bias)
        self.enable_dropout = enable_dropout

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc3(x)

        if self.log_softmax:
            return F.log_softmax(x, dim=-1)
        else:
            return x


def get_mlpnet(dataset: str) -> nn.Module:
    dataset = dataset.lower()
    if dataset == "cifar10":
        return MlpNet(input_size=3072, nh1=40, nh2=20, num_classes=10)
    elif dataset == "mnist":
        return MlpNet(input_size=784, nh1=40, nh2=20, num_classes=10)
    raise NotImplementedError(f"MlpNet is not available for {dataset}")
