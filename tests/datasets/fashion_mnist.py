import os
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
from torchvision.transforms import v2
from tqdm.auto import trange


class TransformedFashionMNIST(datasets.FashionMNIST):

    _TRAIN_FPATH = 'train.pth'
    _TEST_FPATH = 'test.pth'

    def __init__(
            self, root: str, size: int, train: bool = True,
            recompute: bool = False, transforms: list[v2.Transform] = []):
        mnist_root = os.path.join(root, 'fashion_mnist')

        super().__init__(
            mnist_root, train, download=True,
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


_DEFAULT_TRANSFORMS = [
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.2860,), std=(0.3530,))
]


def get_fashion_mnist(
        model_name: str,
        root_dir: str,
        recompute: bool = False) -> tuple[data.Dataset, data.Dataset]:
    model_name = model_name.lower()
    if model_name in ['lenet', 'fullyconnected']:
        train = TransformedFashionMNIST(
            root_dir, size=28, train=True, recompute=recompute,
            transforms=_DEFAULT_TRANSFORMS)
        test = TransformedFashionMNIST(
            root_dir, size=28, train=False, recompute=recompute,
            transforms=_DEFAULT_TRANSFORMS)
        return (train, test)
    raise NotImplementedError(f'FashionMNIST not available for {model_name}')
