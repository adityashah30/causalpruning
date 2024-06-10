import argparse
import datetime
import os

import torch.nn as nn
import torch.optim as optim

import context
from context import get_trainer
from causalpruner import DataConfig, EpochConfig, TrainerConfig, SGDPrunerConfig
from causalpruner import best_device
from models import get_model
from datasets import get_dataset
from pruner.mag_pruner import MagPrunerConfig


def get_optimizer(
        name: str, model: nn.Module, lr: float, momentum: float) -> optim.Optimizer:
    name = name.lower()
    if name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    raise NotImplementedError(f'{name} is not a supported Optimizier')


def main(args):
    now = datetime.datetime.now()
    date = f'{now.year:04d}{now.month:02d}{now.day:02d}'
    time = f'{now.hour:04d}{now.minute:02d}{now.second:02d}'
    identifier = f'{args.pruner}-{date}-{time}'
    train_dataset, test_dataset = get_dataset(
        args.dataset, args.dataset_root_dir, args.recompute_dataset)
    model = get_model(args.model, args.dataset).to(best_device())
    optimizer = get_optimizer(args.optimizer, model, args.lr, args.momentum)
    data_config = DataConfig(
        train_dataset=train_dataset, test_dataset=test_dataset,
        batch_size=args.batch_size, num_workers=args.num_dataset_workers,
        shuffle=args.shuffle_dataset)
    epoch_config = EpochConfig(
        num_pre_prune_epochs=args.num_pre_prune_epochs,
        num_prune_iterations=args.num_prune_iterations,
        num_prune_epochs=args.num_prune_epochs,
        num_post_prune_epochs=args.num_post_prune_epochs)
    trainer_config = TrainerConfig(
        model=model, optimizer=optimizer, data_config=data_config,
        epoch_config=epoch_config, tensorboard_dir=os.path.join(
            args.tensorboard_dir, identifier))
    if args.pruner == 'causalpruner':
        pruner_config = SGDPrunerConfig(
            pruner='SGDPruner', checkpoint_dir=os.path.join(
                args.checkpoint_dir, identifier),
            start_clean=args.start_clean, momentum=args.momentum > 0,
            pruner_lr=args.pruner_lr, prune_threshold=args.prune_threshold,
            l1_regularization_coeff=args.pruner_l1_regularization_coeff,
            causal_weights_num_epochs=args.causal_weights_num_epochs)
    elif args.pruner == 'magpruner':
        pruner_config = MagPrunerConfig(
            pruner='MagPruner', checkpoint_dir=args.checkpoint_dir,
            start_clean=args.start_cean, prune_amount=args.prune_amount_mag)
    trainer = get_trainer(args.pruner, trainer_config, pruner_config)
    trainer.run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Causal Pruning")

    parser.add_argument("--model", type=str,
                        default="lenet", help="Model name")
    parser.add_argument("--dataset", type=str,
                        default="cifar10", help="Dataset name")
    parser.add_argument(
        '--dataset_root_dir', type=str, default='../data',
        help="Directory to download datasets")
    parser.add_argument(
        '--recompute_dataset', type=bool, default=False,
        help="Recomputes dataset transformations if true -- loads existing otherwise")
    parser.add_argument(
        "--num_dataset_workers", type=int, default=2,
        help="Number of dataset workers")
    parser.add_argument(
        '--shuffle_dataset', type=bool, default=True,
        help="Whether to shuffle the train and test datasets")
    parser.add_argument(
        '--tensorboard_dir', type=str, default='../tensorboard',
        help="Directory to write tensorboard data")
    parser.add_argument("--batch_size", type=int,
                        default=8192, help="Batch size")
    parser.add_argument("--num_pre_prune_epochs", type=int, default=10,
                        help="Number of epochs for pretraining")
    parser.add_argument("--num_prune_iterations", type=int, default=10,
                        help="Number of iterations to prune")
    parser.add_argument("--num_prune_epochs", type=int, default=10,
                        help="Number of epochs for pruning")
    parser.add_argument("--num_post_prune_epochs", type=int, default=100,
                        help="Number of epochs to run post training")
    parser.add_argument("--optimizer", type=str,
                        default="sgd", help="Optimizier", choices=["sgd"])
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.0, help="Momentum")
    parser.add_argument(
        "--pruner", type=str, default="causalpruner",
        help="Method for pruning", choices=["causalpruner", "magpruner"])
    parser.add_argument(
        '--checkpoint_dir', type=str, default='../checkpoints',
        help="Checkpoint dir to write model weights and losses")
    parser.add_argument(
        '--start_clean', type=bool, default=True,
        help="Controls if the pruner deletes any existing directories when starting")
    parser.add_argument(
        '--pruner_lr', type=float, default=1e-3,
        help="Learning rate for causal pruner")
    parser.add_argument(
        '--prune_threshold', type=float, default=5e-6,
        help="Weight threshold for causal pruner below which the weights are made zero")
    parser.add_argument(
        '--pruner_l1_regularization_coeff', type=float, default=1e-5,
        help="Causal Pruner L1 regularization coefficient")
    parser.add_argument(
        '--causal_weights_num_epochs', type=int, default=500,
        help="Number of epochs to run causal pruner training")
    parser.add_argument("--prune_amount_mag", type=float,
                        default=0.4, help="Amount")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
