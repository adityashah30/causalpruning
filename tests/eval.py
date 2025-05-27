# autopep8: off
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# autopep: on

import argparse
import os
from tqdm.auto import tqdm

from lightning.fabric import Fabric
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics

from causalpruner import Pruner
from models import get_model
from datasets import get_dataset
from test_utils import (
    EvalMetrics,
    MetricsComputer,
)

torch.set_float32_matmul_precision("medium")


@torch.no_grad
def load_model(fabric: Fabric, model: nn.Module, path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        fabric.load(path, {"model": model})
    except:
        Pruner.apply_identity_masks_to_model(model)
        fabric.load(path, {"model": model})
    return True


@torch.no_grad
def eval_model(
    fabric: Fabric, model: nn.Module, testloader: DataLoader, num_classes: int
) -> tuple[float, EvalMetrics]:
    model.eval()
    metrics_computer = MetricsComputer(num_classes).to(fabric.device)
    val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(
        fabric.device
    )
    for data in tqdm(testloader, leave=False):
        inputs, labels = data
        outputs = model(inputs)
        metrics_computer.add(outputs, labels)
        val_accuracy(outputs, labels)
    return (val_accuracy.compute(), metrics_computer.compute())


@torch.no_grad
def print_prune_stats(model: nn.Module):
    tqdm.write("\n======================================================\n")
    all_params_total = 0
    all_params_pruned = 0
    for name, param in model.named_buffers():
        name = name.rstrip(".weight_mask")
        non_zero = torch.count_nonzero(param)
        total = torch.count_nonzero(torch.ones_like(param))
        all_params_total += total
        pruned = total - non_zero
        all_params_pruned += pruned
        percent = 100 * pruned / total
        tqdm.write(
            f"Name: {name}; Total: {total}; "
            f"non-zero: {non_zero}; pruned: {pruned}; "
            f"percent: {percent:.2f}%"
        )
    all_params_non_zero = all_params_total - all_params_pruned
    all_params_percent = 100 * all_params_pruned / (all_params_total + 1e-6)
    tqdm.write(
        f"Name: All; Total: {all_params_total}; "
        f"non-zero: {all_params_non_zero}; "
        f"pruned: {all_params_pruned}; "
        f"percent: {all_params_percent:.2f}%"
    )
    tqdm.write("\n======================================================\n")


def main(args: argparse.Namespace):
    print(args)

    fabric = Fabric(devices=args.device_ids, accelerator="auto")
    fabric.launch()

    model_name = args.model
    dataset_name = args.dataset
    dataset_root_dir = args.dataset_root_dir
    model_checkpoint = args.model_checkpoint

    _, test_dataset, num_classes = get_dataset(
        dataset_name, model_name, data_root_dir=dataset_root_dir
    )
    testloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_dataset_workers,
        persistent_workers=args.num_dataset_workers > 0,
    )
    testloader = fabric.setup_dataloaders(testloader)

    model = get_model(model_name, dataset_name)
    model = fabric.setup(model)
    if not load_model(fabric, model, model_checkpoint):
        tqdm.write(f"Model does not exist at {model_checkpoint}")
        return
    tqdm.write(f"Model loaded from {model_checkpoint}")

    val_accuracy, eval_metrics = eval_model(fabric, model, testloader, num_classes)
    val_accuracy *= 100

    if not fabric.is_global_zero:
        return
    tqdm.write(f"Model: {model_name}")
    tqdm.write(f"Dataset: {dataset_name}")
    print_prune_stats(model)
    tqdm.write("\n======================================================\n")
    tqdm.write("Final eval metrics:")
    tqdm.write(f"Total Accuracy: {val_accuracy:.2f}%")
    tqdm.write(f"Accuracy: {eval_metrics.accuracy}")
    tqdm.write(f"Precision: {eval_metrics.precision}")
    tqdm.write(f"Recall: {eval_metrics.recall}")
    tqdm.write(f"F1 Score: {eval_metrics.f1_score}")
    tqdm.write("\n======================================================\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot pruning graphs")

    parser.add_argument(
        "--model",
        type=str,
        choices=["lenet", "alexnet", "resnet18", "resnet20", "resnet50"],
        help="Model name",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "fashionmnist", "imagenet", "tinyimagenet"],
        help="Dataset name",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="",
        help="Path to model checkpoint. Loads the checkpoint if model is given -- else uses the default version",
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default="../data",
        help="Directory to download datasets",
    )
    parser.add_argument(
        "--device_ids",
        type=str,
        default="-1",
        help="The device id. Useful for multi device systems",
    )
    parser.add_argument(
        "--num_dataset_workers", type=int, default=4, help="Number of dataset workers"
    )
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
