import torch
import torch.utils.data as data
from torchvision.models import ResNet50_Weights
from torchvision.datasets import ImageNet


@torch.no_grad
def get_imagenet(
        model_name: str,
        root_dir: str) -> tuple[data.Dataset, data.Dataset]:
    model_name = model_name.lower()
    if model_name == 'resnet50':
        transforms = ResNet50_Weights.IMAGENET1K_V2.transforms
        train = ImageNet(root=root_dir, split='train', transform=transforms)
        test = ImageNet(root=root_dir, split='test', transforms=transforms)
        return (train, test)
    raise NotImplementedError(f'CIFAR10 not available for {model_name}')
