import os
import torch
import torch.utils.data as data
from torchvision.datasets import ImageNet
from torchvision.transforms import v2


@torch.no_grad
def get_imagenet(
        model_name: str,
        root_dir: str) -> tuple[data.Dataset, data.Dataset]:
    model_name = model_name.lower()
    imagenet_root = os.path.join(root_dir, 'imagenet')
    if model_name == 'resnet50':
        transforms = v2.Compose([
            v2.RandomResizedCrop(224),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])
        train = ImageNet(root=imagenet_root, split='train',
                         transform=transforms)
        test = ImageNet(root=imagenet_root, split='val',
                        transform=transforms)
        return (train, test)
    raise NotImplementedError(f'CIFAR10 not available for {model_name}')
