import os
import torch
import torch.utils.data as data
from torchvision.datasets import ImageNet
from torchvision.transforms import v2


@torch.no_grad
def get_imagenet(
        model_name: str,
        data_root_dir: str) -> tuple[data.Dataset, data.Dataset]:
    model_name = model_name.lower()
    imagenet_root = os.path.join(data_root_dir, 'imagenet')
    if model_name == 'resnet50':
        default_transforms = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])]
        train_transforms = v2.Compose([
            v2.RandomResizedCrop(224),
            v2.RandomHorizontalFlip(),
        ] + default_transforms)
        test_transforms = v2.Compose(
            [v2.Resize((224, 224))] + default_transforms)
        train = ImageNet(root=imagenet_root, split='train',
                         transform=train_transforms)
        test = ImageNet(root=imagenet_root, split='val',
                        transform=test_transforms)
        return (train, test)
    raise NotImplementedError(f'CIFAR10 not available for {model_name}')
