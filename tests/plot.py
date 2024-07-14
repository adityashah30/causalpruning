# autopep8: off
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# autopep: on

import argparse
import os

import pandas as pd
from tbparse import SummaryReader

from models import get_model


def get_accuracy_from_tb(df: pd.DataFrame) -> float:
    accuracy = df[df.tag == 'Accuracy/Test'].value
    return max(accuracy)


def get_pruned_percent_from_tb(df: pd.DataFrame) -> float:
    return df[df.tag == 'all/pruned_percent'].value.iat[-1]


def main(args: argparse.Namespace):
    if args.verbose:
        print(args)
    root_dir = args.tensorboard_root_dir
    model = args.model
    dataset = args.dataset
    momentum = args.momentum

    num_params = sum(
        map(lambda l: l.numel(), get_model(model, dataset).parameters()))
    print(num_params)

    # No Prune
    no_prune_tb_dir = os.path.join(
            root_dir, f'noprune_{model}_{dataset}_{momentum}')
    df = SummaryReader(no_prune_tb_dir).scalars
    accuracy = get_accuracy_from_tb(df)
    print(f'NoPrune: {accuracy}')

    # Causal Pruning
    causal_pruning_tb_dir = os.path.join(
            root_dir, f'causalpruner_{model}_{dataset}_{momentum}')
    df = SummaryReader(causal_pruning_tb_dir).scalars
    accuracy = get_accuracy_from_tb(df)
    pruned_percent = get_pruned_percent_from_tb(df)
    print(f'CausalPruning: {accuracy}; Pruned Percent: {pruned_percent:.4f}%')

    # Mag Pruning
    mag_pruning_tb_dir = os.path.join(
            root_dir, f'magpruner_{model}_{dataset}_{momentum}')
    df = SummaryReader(mag_pruning_tb_dir).scalars
    accuracy = get_accuracy_from_tb(df)
    pruned_percent = get_pruned_percent_from_tb(df)
    print(f'MagPruning: {accuracy}; Pruned Percent: {pruned_percent:.4f}%')



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot pruning graphs')

    parser.add_argument('--tensorboard_root_dir', type=str,
                        default='../tensorboard', help='Root tensorboard dir')
    parser.add_argument('--model', type=str,
                        choices=['alexnet', 'lenet', 'fullyconnected'])
    parser.add_argument('--dataset', type=str,
                        choices=['cifar10', 'fashionmnist'],
                        default='cifar10', help='Dataset name')
    parser.add_argument('--momentum', type=float,
                        default=0.0, help='Model training momentum')
    parser.add_argument(
        '--verbose', action=argparse.BooleanOptionalAction,
        default=True, help='Output verbosity')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
