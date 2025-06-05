from concurrent.futures import ThreadPoolExecutor
import io
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import v2, AutoAugmentPolicy
from torchvision.datasets import ImageNet
import torch.utils.data as data
import torch
from typing import Any, Optional
import os
import torch.multiprocessing as multiprocessing


class ImageNetMemory(ImageNet):
    _manager = None

    @classmethod
    def manager(cls) -> multiprocessing.Manager:
        if cls._manager is None:
            cls._manager = multiprocessing.Manager()
        return cls._manager

    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
    ):
        super().__init__(root, split)
        self.user_transform = transform
        self.user_target_transform = target_transform
        self._load_data_in_memory()

    def _load_data_in_memory(self):
        num_samples = len(self.samples)
        self._data = ImageNetMemory.manager().list([None] * num_samples)
        max_workers = 16

        def load_file(start_idx, end_idx, data, samples):
            for idx in range(start_idx, end_idx):
                path, target = samples[idx]
                with open(path, "rb") as f:
                    data[idx] = (f.read(), target)

        num_items_per_worker = (num_samples + max_workers - 1) // max_workers
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index in range(max_workers):
                start_idx = index * num_items_per_worker
                end_idx = start_idx + num_items_per_worker
                if end_idx > num_samples:
                    end_idx = num_samples
                executor.submit(
                    load_file,
                    start_idx,
                    end_idx,
                    self._data,
                    self.samples,
                )

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        input, target = self._data[index]
        input = Image.open(io.BytesIO(input)).convert("RGB")
        if self.user_transform is not None:
            input = self.user_transform(input)
        if self.user_target_transform is not None:
            target = self.user_target_transform(target)
        return (input, target)


@torch.no_grad()
def get_imagenet(
    model_name: str,
    data_root_dir: str,
    in_memory: bool = False,
) -> tuple[data.Dataset, data.Dataset]:
    model_name = model_name.lower()
    imagenet_root = os.path.join(data_root_dir, "imagenet")
    if model_name in [
        "mobilenet_trained",
        "mobilenet_untrained",
        "resnet50_torch",
        "resnet50_untrained",
        "resnet50_trained",
    ]:
        interpolation = InterpolationMode.BILINEAR
        # train_transforms = v2.Compose(
        #     [
        #         v2.RandomResizedCrop(224, interpolation=interpolation, antialias=True),
        #         v2.AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
        #         # v2.RandomHorizontalFlip(0.5),
        #         # v2.TrivialAugmentWide(interpolation=interpolation),
        #         v2.PILToTensor(),
        #         v2.ToDtype(torch.float, scale=True),
        #         v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #         # v2.RandomErasing(0.1),
        #         v2.ToPureTensor(),
        #     ]
        # )
        # test_transforms = v2.Compose(
        #     [
        #         v2.Resize(256, interpolation=interpolation, antialias=True),
        #         v2.CenterCrop(224),
        #         v2.PILToTensor(),
        #         v2.ToDtype(torch.float, scale=True),
        #         v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #         v2.ToPureTensor(),
        #     ]
        # )
        train_transforms = v2.Compose(
            [
                v2.RandomResizedCrop(224),
                v2.RandomHorizontalFlip(),
                v2.PILToTensor(),
                v2.ToDtype(torch.float, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                v2.ToPureTensor(),
            ]
        )
        test_transforms = v2.Compose(
            [
                v2.Resize(256),
                v2.CenterCrop(224),
                v2.PILToTensor(),
                v2.ToDtype(torch.float, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                v2.ToPureTensor(),
            ]
        )
        if in_memory:
            train = ImageNetMemory(
                root=imagenet_root, split="train", transform=train_transforms
            )
            test = ImageNetMemory(
                root=imagenet_root, split="val", transform=test_transforms
            )
        else:
            train = ImageNet(
                root=imagenet_root, split="train", transform=train_transforms
            )
            test = ImageNet(root=imagenet_root, split="val", transform=test_transforms)
        return (train, test)
    raise NotImplementedError(f"Imagenet not available for {model_name}")
