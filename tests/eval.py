# autopep8: off
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# autopep: on

import argparse
import os
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy

from causalpruner import (
    Pruner,
    best_device,
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
def eval_model(model: nn.Module, 
               testloader: DataLoader, 
               device: torch.device) -> float:
    model.eval()
    accuracy = MulticlassAccuracy().to(device)
    for data in tqdm(testloader):
        inputs, labels = data
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(inputs)
        accuracy.update(outputs, labels)
    accuracy = accuracy.compute().item()
    return accuracy


@torch.no_grad
def print_prune_stats(model: nn.Module):
    tqdm.write('\n======================================================\n')
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
        tqdm.write(f'Name: {name}; Total: {total}; '
                  f'non-zero: {non_zero}; pruned: {pruned}; '
                  f'percent: {percent:.2f}%')
    all_params_non_zero = all_params_total - all_params_pruned
    all_params_percent = 100 * all_params_pruned / \
        (all_params_total + 1e-6)
    tqdm.write(f'Name: All; Total: {all_params_total}; '
               f'non-zero: {all_params_non_zero}; '
               f'pruned: {all_params_pruned}; '
               f'percent: {all_params_percent:.2f}%')
    tqdm.write('\n======================================================\n')


def main(args: argparse.Namespace):
    print(args)
    model_name = args.model
    dataset_name = args.dataset
    dataset_root_dir = args.dataset_root_dir
    model_checkpoint = args.model_checkpoint
    device = best_device(args.device_id)

    model = get_model(model_name, dataset_name)
    if model_checkpoint != '':
        load_model(model, model_checkpoint)
    model = model.to(device)

    _, test_dataset, _ = get_dataset(
        dataset_name, model_name, root_dir=dataset_root_dir)
    testloader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=args.shuffle, pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0)

    accuracy = eval_model(model, testloader, device)

    tqdm.write(f'Model: {model_name}')
    tqdm.write(f'Dataset: {dataset_name}')
    print_prune_stats(model)
    tqdm.write(f'Accuracy: {accuracy}')


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
    parser.add_argument('--device_id',
                        type=int,
                        default=0,
                        help='The device id. Useful for multi device systems')
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of dataset workers')
    parser.add_argument(
        '--shuffle', action=argparse.BooleanOptionalAction,
        default=True,
        help='Whether to shuffle the test datasets')
    parser.add_argument('--batch_size', type=int,
                        default=256, help='Batch size')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
