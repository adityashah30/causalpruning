import os
from tests.datasets import get_dataset


ROOT_DIR = os.path.expanduser('.')
DATA_ROOT_DIR = os.path.join(ROOT_DIR, 'data')


if __name__ == '__main__':
    print('Downloading Fashion-MNIST')
    _ = get_dataset('fashionmnist', 'lenet', DATA_ROOT_DIR)
    print('Downloading CIFAR10')
    _ = get_dataset('cifar10', 'lenet', DATA_ROOT_DIR)
