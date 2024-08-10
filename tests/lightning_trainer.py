# autopep8: off
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# autopep: on

import argparse
import os
from tqdm.auto import tqdm, trange

from lightning.fabric import Fabric
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy

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


def train_model(fabric: Fabric,
                model: nn.Module,
                optimizer: optim.Optimizer,
                trainloader: DataLoader,
                testloader: DataLoader,
                num_epochs: int):
    print('Training model')

    for epoch in (pbar := trange(num_epochs, leave=False)):
        model.train()
        for (inputs, labels) in tqdm(trainloader, leave=False):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            fabric.backward(loss)
            optimizer.step()
        accuracy = eval_model(fabric, model, testloader)
        pbar.set_description(f'Epoch: {epoch + 1}; Accuracy: {accuracy:.4f}')


@torch.no_grad
def eval_model(fabric: Fabric,
               model: nn.Module,
               testloader: DataLoader) -> float:
    model.eval()
    accuracy = MulticlassAccuracy().to(fabric.device)
    for data in tqdm(testloader, leave=False):
        inputs, labels = data
        outputs = model(inputs)
        accuracy.update(outputs, labels)
    accuracy = accuracy.compute().item()
    return accuracy


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


def main(args: argparse.Namespace):
    print(args)

    fabric = Fabric(devices=args.device_ids, accelerator='auto')
    fabric.launch()

    model_name = args.model
    dataset_name = args.dataset
    dataset_root_dir = args.dataset_root_dir
    model_checkpoint = args.model_checkpoint

    model = get_model(model_name, dataset_name)
    if model_checkpoint != '':
        load_model(model, model_checkpoint)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    num_workers = args.num_dataset_workers
    persistent_workers = num_workers > 0

    train_dataset, test_dataset, _ = get_dataset(
        dataset_name, model_name, root_dir=dataset_root_dir)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                 shuffle=args.shuffle, pin_memory=True, num_workers=num_workers,
                                 persistent_workers=persistent_workers)
    testloader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, pin_memory=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers)

    model, optimizer = fabric.setup(model, optimizer)
    trainloader, testloader = fabric.setup_dataloaders(trainloader, testloader)

    if args.train:
        train_model(fabric,
                    model,
                    optimizer,
                    trainloader,
                    testloader,
                    args.num_train_epochs)
        if args.trained_model_checkpoint != '':
            fabric.save(args.trained_model_checkpoint, model.state_dict())

    accuracy = eval_model(fabric, model, testloader)

    print(f'Model: {model_name}')
    print(f'Dataset: {dataset_name}')
    print_prune_stats(model)
    print(f'Accuracy: {accuracy:.6f}')


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
                        nargs='+',
                        type=int,
                        default=-1,
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
    parser.add_argument(
        '--train', action=argparse.BooleanOptionalAction,
        default=False,
        help='Whether to train the model before evaling')
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
