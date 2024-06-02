import os
import torch
import torchvision.datasets as datasets

from torchvision.transforms import v2


class TransformedCIFAR10(datasets.CIFAR10):

    _TRAIN_FPATH = 'train.pth'
    _TEST_FPATH = 'test.pth'
    _TRANSFORM_TRAIN = v2.Compose(
        [v2.RandomCrop(32, padding=4),
         v2.RandomHorizontalFlip(),
         v2.ToImage(),
         v2.ToDtype(torch.float32, scale=True),
         v2.Normalize(
            (0.4914, 0.4822, 0.4465),
             (0.2023, 0.1994, 0.2010))])
    _TRANSFORM_TEST = v2.Compose(
        [v2.ToImage(),
         v2.ToDtype(torch.float32, scale=True),
         v2.Normalize(
            (0.4914, 0.4822, 0.4465),
             (0.2023, 0.1994, 0.2010))])

    def __init__(self, root: str, train: bool = True, recompute: bool = False):
        super().__init__(
            root, train, download=True,
            transform=self._TRANSFORM_TRAIN
            if train else self._TRANSFORM_TEST)

        file_path = self._TRAIN_FPATH if train else self._TEST_FPATH
        self.fpath = os.path.join(self.root, file_path)

        if recompute or not os.path.exists(self.fpath):
            self.transform_and_save()
        else:
            print('Data already transformed and saved')

        self.load_data()

    def transform_and_save(self):
        data = []
        targets = []
        for idx in range(super().__len__()):
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
