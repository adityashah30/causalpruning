import os
import torch
import torchvision.datasets as datasets

from torchvision.transforms import v2
from tqdm.auto import trange


class TransformedCIFAR10(datasets.CIFAR10):

    _TRAIN_FPATH = 'train.pth'
    _TEST_FPATH = 'test.pth'

    def __init__(
            self, root: str, size: int, train: bool = True,
            recompute: bool = False, transforms: list[v2.Transform] = []):
        cifar10_root = os.path.join(root, 'cifar10')

        super().__init__(
            cifar10_root, train, download=True,
            transform=v2.Compose(transforms))

        size_dir = os.path.join(self.root, f'{size}x{size}')
        os.makedirs(size_dir, exist_ok=True)

        file_path = self._TRAIN_FPATH if train else self._TEST_FPATH
        self.fpath = os.path.join(size_dir, file_path)

        if recompute or not os.path.exists(self.fpath):
            self.transform_and_save()
        else:
            print('Data already transformed and saved')

        self.load_data()

    def transform_and_save(self):
        data = []
        targets = []
        for idx in trange(super().__len__()):
            datum, target = super().__getitem__(idx)
            data.append(datum)
            targets.append(target)
        data = torch.stack(data, dim=0)
        targets = torch.tensor(targets)
        torch.save({'data': data, 'targets': targets}, self.fpath)

    def load_data(self):
        dict = torch.load(self.fpath)
        self.data = dict['data']
        self.targets = dict['targets']

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]

    def __len__(self) -> int:
        return len(self.data)
