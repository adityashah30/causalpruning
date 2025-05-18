import os
import torch
import torch.utils.data as data
from torchvision.datasets import ImageNet
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode


@torch.no_grad
def get_imagenet(
    model_name: str, data_root_dir: str
) -> tuple[data.Dataset, data.Dataset]:
    model_name = model_name.lower()
    imagenet_root = os.path.join(data_root_dir, "imagenet")
    if model_name in ["resnet50", "resnet50_untrained"]:
        interpolation = InterpolationMode.BILINEAR
        train_transforms = v2.Compose(
            [
                v2.RandomResizedCrop(
                    224, interpolation_mode=interpolation, antialias=True
                ),
                v2.RandomHorizontalFlip(0.5),
                v2.TrivialAugmentWide(interpolation=interpolation),
                v2.PILToTensor(),
                v2.ToDtype(torch.float, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
                v2.RandomErasing(0.1),
                v2.ToPureTensor(),
            ]
        )
        test_transforms = v2.Compose(
            [
                v2.Resize(256, interpolation=interpolation, antialias=True),
                v2.CenterCrop(224),
                v2.PILToTensor(),
                v2.ToDtype(torch.float, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
                v2.ToPureTensor(),
            ]
        )
        train = ImageNet(root=imagenet_root, split="train",
                         transform=train_transforms)
        test = ImageNet(root=imagenet_root, split="val",
                        transform=test_transforms)
        return (train, test)
    raise NotImplementedError(f"Imagenet not available for {model_name}")
