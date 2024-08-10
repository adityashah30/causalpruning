# autopep8: off
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# autopep: on

import argparse

import lightning.pytorch as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchmetrics

from causalpruner import (
    Pruner,
)
from models import get_model
from datasets import get_dataset

@torch.no_grad
def load_model(model: nn.Module, path: str):
    if not os.path.exists(path):
        print(f'Model not found at {path}')
        return
    Pruner.apply_identity_masks_to_model(model)
    print(f'Model loaded from {path}')
    model.load_state_dict(torch.load(path))


@torch.no_grad
def print_prune_stats(model: nn.Module):
    print('======================================================')
    all_params_total = 0
    all_params_pruned = 0
    for (name, param) in model.named_buffers():
        name = name.rstrip('.weight_mask')
        non_zero = torch.count_nonzero(param)
        total = torch.count_nonzero(torch.ones_like(param))
        all_params_total += total
        pruned = total - non_zero
        all_params_pruned += pruned
        percent = 100 * pruned / total
        print(f'Name: {name}; Total: {total}; '
                  f'non-zero: {non_zero}; pruned: {pruned}; '
                  f'percent: {percent:.2f}%')
    all_params_non_zero = all_params_total - all_params_pruned
    all_params_percent = 100 * all_params_pruned / \
        (all_params_total + 1e-6)
    print(f'Name: All; Total: {all_params_total}; '
          f'non-zero: {all_params_non_zero}; '
          f'pruned: {all_params_pruned}; '
          f'percent: {all_params_percent:.2f}%')
    print('======================================================')


class ModuleLightningModule(L.LightningModule):

    def __init__(self, model: nn.Module, num_classes: int, lr: float):
        super().__init__()
        self.model = model
        self.lr = lr
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes)

    def load_model(self, path):
        load_model(self.model, path)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def training_step(
            self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
            self, batch: torch.Tensor, batch_idx: int):
        inputs, labels = batch
        outputs = self(inputs)
        self.accuracy(outputs, labels)
        self.log('val_acc', self.accuracy, on_epoch=True, prog_bar=True)


def main(args: argparse.Namespace):
    print(args)

    model_name = args.model
    dataset_name = args.dataset
    dataset_root_dir = args.dataset_root_dir
    model_checkpoint = args.model_checkpoint

    devices = [0]
    if args.device_ids != '':
        devices = list(
            map(lambda a: int(a.strip()), args.device_ids.split(',')))

    print(f'Model: {model_name}')
    print(f'Dataset: {dataset_name}')

    num_workers = args.num_dataset_workers
    persistent_workers = num_workers > 0

    train_dataset, test_dataset, num_classes = get_dataset(
        dataset_name, model_name, root_dir=dataset_root_dir)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size,
                             pin_memory=True, num_workers=num_workers,
                             persistent_workers=persistent_workers)
    testloader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        pin_memory=True, num_workers=num_workers,
        persistent_workers=persistent_workers)

    model = get_model(model_name, dataset_name)
    lightning_module = ModuleLightningModule(model, num_classes, args.lr)
    if model_checkpoint != '':
        lightning_module.load_model(model_checkpoint)

    trainer = L.Trainer(devices=devices, 
                        max_epochs=args.num_train_epochs)
    trainer.fit(
        lightning_module,
        train_dataloaders=trainloader,
        val_dataloaders=testloader)

    trainer.validate(lightning_module, dataloaders=testloader)
    print_prune_stats(model)

    torch.save(lightning_module.model.state_dict(), 
               args.trained_model_checkpoint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot pruning graphs')

    parser.add_argument('--model', type=str,
                        choices=['lenet', 'alexnet', 'resnet18', 'resnet50'],
                        help='Model name')
    parser.add_argument('--dataset', type=str,
                        choices=['cifar10', 'fashionmnist', 'imagenet'],
                        help='Dataset name')
    parser.add_argument('--model_checkpoint',
                        type=str, default='',
                        help='Path to model checkpoint. Loads the checkpoint if model is given -- else uses the default version')
    parser.add_argument(
        '--dataset_root_dir', type=str, default='../data',
        help='Directory to download datasets')
    parser.add_argument('--device_ids',
                        type=str,
                        default='',
                        help='The device ids. Useful for multi device systems')
    parser.add_argument(
        '--num_dataset_workers', type=int, default=4,
        help='Number of dataset workers')
    parser.add_argument(
        '--shuffle', action=argparse.BooleanOptionalAction,
        default=True,
        help='Whether to shuffle the test datasets')
    parser.add_argument('--batch_size', type=int,
                        default=256, help='Batch size')
    parser.add_argument('--trained_model_checkpoint',
                        type=str, default='',
                        help='Path to write trained model checkpoint. Used if not empty')
    parser.add_argument('--num_train_epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Training optimizer learning rate')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
