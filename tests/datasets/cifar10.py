import os
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
from torchvision.transforms import v2
from tqdm.auto import tqdm


class TransformedCIFAR10(datasets.CIFAR10):

    _TRAIN_FPATH = 'train.pth'
    _TRAIN_SHARDS = 10
    _TEST_FPATH = 'test.pth'
    _TEST_SHARDS = 2

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
        self.num_shards = self._TRAIN_SHARDS if train else self._TEST_SHARDS

        if recompute or not self._exists_data():
            self.transform_and_save()
        else:
            print('Data already transformed and saved')

        self.load_data()

    def _exists_data(self) -> bool:
        for shard_idx in range(self.num_shards):
            path = self.fpath + f'.{shard_idx}'
            if not os.path.exists(path):
                return False
        return True

    @torch.no_grad
    def transform_and_save(self):
        num_items = super().__len__()
        items_per_shard = (num_items + self.num_shards - 1) // self.num_shards
        pbar = tqdm(total=num_items)
        for start_idx in range(0, num_items, items_per_shard):
            data = []
            targets = []
            end_idx = start_idx + items_per_shard
            for idx in range(start_idx, end_idx):
                pbar.update(1)
                datum, target = super().__getitem__(idx)
                data.append(datum)
                targets.append(target)
            data = torch.stack(data, dim=0)
            targets = torch.tensor(targets)
            shard_index = start_idx // items_per_shard
            shard_path = self.fpath + f'.{shard_index}'
            torch.save({'data': data, 'targets': targets}, shard_path)

    @torch.no_grad
    def load_data(self,):
        self.shards_data = []
        for shard_idx in range(self.num_shards):
            fpath = self.fpath + f'.{shard_idx}'
            dict = torch.load(fpath)
            self.shards_data.append(dict)

    @torch.no_grad
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        num_items = super().__len__()
        assert idx <= num_items
        items_per_shard = (
            num_items + self.num_shards - 1) // self.num_shards
        shard_idx = idx // items_per_shard
        idx_in_shard = idx % items_per_shard
        shard_data = self.shards_data[shard_idx]
        data = shard_data['data']
        targets = shard_data['targets']
        return data[idx_in_shard], targets[idx_in_shard]


_DEFAULT_TRAIN_TRANSFORMS = [
    v2.RandomHorizontalFlip(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010))
]
_DEFAULT_TEST_TRANSFORMS = [
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010))
]


@torch.no_grad
def get_cifar_10(
        model_name: str,
        root_dir: str,
        recompute: bool = False) -> tuple[data.Dataset, data.Dataset]:

    model_name = model_name.lower()
    if model_name in ['lenet', 'fullyconnected']:
        train = TransformedCIFAR10(
            root_dir, size=32, train=True, recompute=recompute,
            transforms=_DEFAULT_TRAIN_TRANSFORMS)
        test = TransformedCIFAR10(
            root_dir, size=32, train=False, recompute=recompute,
            transforms=_DEFAULT_TEST_TRANSFORMS)
        return (train, test)
    elif model_name == 'alexnet':
        train = TransformedCIFAR10(
            root_dir, size=227, train=True, recompute=recompute,
            transforms=[v2.Resize((227, 227))] + _DEFAULT_TRAIN_TRANSFORMS)
        test = TransformedCIFAR10(
            root_dir, size=227, train=False, recompute=recompute,
            transforms=[v2.Resize((227, 227))] + _DEFAULT_TEST_TRANSFORMS)
        return (train, test)
    elif model_name == 'resnet18':
        train = TransformedCIFAR10(
            root_dir, size=224, train=True, recompute=recompute,
            transforms=[v2.Resize((224, 224))] + _DEFAULT_TRAIN_TRANSFORMS)
        test = TransformedCIFAR10(
            root_dir, size=224, train=False, recompute=recompute,
            transforms=[v2.Resize((224, 224))] + _DEFAULT_TEST_TRANSFORMS)
        return (train, test)
    raise NotImplementedError(f'CIFAR10 not available for {model_name}')
