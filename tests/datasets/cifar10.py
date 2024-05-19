import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def get_cifar10(root_dir: str, download: bool = True) -> tuple[data.Dataset, data.Dataset]:
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ]
    )
    trainset = datasets.CIFAR10(
        root=root_dir, train=True, download=download, transform=transform_train
    )
    testset = datasets.CIFAR10(
        root=root_dir, train=False, download=download, transform=transform_test
    )
    return trainset, testset
