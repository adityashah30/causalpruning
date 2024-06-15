# autopep8: off
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# autopep: on

import argparse
from datetime import datetime

import torch.nn as nn
import torch.optim as optim

from causalpruner import (
    CausalWeightsTrainerConfig,
    Pruner,
    PrunerConfig,
    SGDPruner,
    SGDPrunerConfig,
    best_device,
)
from models import get_model
from datasets import get_dataset
from pruner.mag_pruner import (
    MagPruner, 
    MagPrunerConfig,
)
from trainer import (
    DataConfig, 
    EpochConfig,
    Pruner,
    Trainer,
    TrainerConfig,
)


def get_optimizer(
        name: str, model: nn.Module, lr: float, momentum: float) -> optim.Optimizer:
    name = name.lower()
    if name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    raise NotImplementedError(f'{name} is not a supported Optimizier')


def get_post_prune_optimizer(
        name: str, model: nn.Module, lr: float) -> optim.Optimizer:
    name = name.lower()
    if name == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr)
    raise NotImplementedError(
        f'{name} is not a supported post-prune Optimizier')


def get_pruner(pruner_config: PrunerConfig) -> Pruner:
    if isinstance(pruner_config, SGDPrunerConfig):
        return SGDPruner(pruner_config)
    elif isinstance(pruner_config, MagPrunerConfig):
        return MagPruner(pruner_config)
    raise NotImplementedError(f'{type(pruner_config)} is not supported yet')


def main(args):
    if args.verbose:
        print(args)
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    pruner = args.pruner
    model_name = args.model
    dataset_name = args.dataset
    identifier = f'{pruner}_{model_name}_{dataset_name}_{timestamp}'
    checkpoint_dir = os.path.join(args.checkpoint_dir, identifier)
    tensorboard_dir = os.path.join(args.tensorboard_dir, identifier)
    train_dataset, test_dataset = get_dataset(
        dataset_name, model_name, args.dataset_root_dir, args.recompute_dataset)
    model = get_model(model_name, dataset_name).to(best_device())
    optimizer = get_optimizer(args.optimizer, model, args.lr, args.momentum)
    post_prune_optimizer = get_post_prune_optimizer(args.post_prune_optimizer,
                                                    model,
                                                    args.post_prune_lr)
    data_config = DataConfig(
        train_dataset=train_dataset, test_dataset=test_dataset,
        batch_size=args.batch_size, num_workers=args.num_dataset_workers,
        shuffle=args.shuffle_dataset, pin_memory=args.pin_memory)
    epoch_config = EpochConfig(
        num_pre_prune_epochs=args.num_pre_prune_epochs,
        num_prune_iterations=args.num_prune_iterations,
        num_prune_epochs=args.num_prune_epochs,
        num_post_prune_epochs=args.num_post_prune_epochs)
    trainer_config = TrainerConfig(
        model=model, optimizer=optimizer,
        post_prune_optimizer=post_prune_optimizer, data_config=data_config,
        epoch_config=epoch_config, tensorboard_dir=tensorboard_dir,
        checkpoint_dir=checkpoint_dir, verbose=args.verbose)
    if args.pruner == 'causalpruner':
        causal_weights_trainer_config = CausalWeightsTrainerConfig(
            init_lr=args.causal_pruner_init_lr, 
            momentum=args.momentum > 0,
            l1_regularization_coeff=args.causal_pruner_l1_regularization_coeff,
            max_iter=args.causal_pruner_max_iter, 
            loss_tol=args.causal_pruner_loss_tol, 
            num_iter_no_change=args.causal_pruner_num_iter_no_change,
            backend=args.causal_pruner_backend)
        pruner_config = SGDPrunerConfig(
            model=model, 
            pruner='SGDPruner', 
            checkpoint_dir=checkpoint_dir,
            start_clean=args.start_clean, 
            trainer_config=causal_weights_trainer_config, 
            device=best_device())
    elif args.pruner == 'magpruner':
        pruner_config = MagPrunerConfig(
            model=model,
            pruner='MagPruner', 
            checkpoint_dir=checkpoint_dir,
            start_clean=args.start_clean, 
            prune_amount=args.mag_pruner_amount, 
            device=best_device())
    pruner = get_pruner(pruner_config)
    trainer = Trainer(trainer_config, pruner)
    trainer.run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Causal Pruning')

    parser.add_argument('--model', type=str, choices=['alexnet', 'lenet'],
                        default='lenet', help='Model name')
    parser.add_argument('--dataset', type=str,
                        default='cifar10', help='Dataset name')
    parser.add_argument(
        '--dataset_root_dir', type=str, default='../data',
        help='Directory to download datasets')
    parser.add_argument(
        '--recompute_dataset', type=bool, default=False,
        help='Recomputes dataset transformations if true -- loads existing otherwise')
    parser.add_argument(
        '--num_dataset_workers', type=int, default=0,
        help='Number of dataset workers')
    parser.add_argument(
        '--shuffle_dataset', type=bool, default=True,
        help='Whether to shuffle the train and test datasets')
    parser.add_argument(
        '--pin_memory', type=bool, default=True,
        help='Whether to pin the Dataloader memory for train and test datasets')
    parser.add_argument(
        '--tensorboard_dir', type=str, default='../tensorboard',
        help='Directory to write tensorboard data')
    parser.add_argument('--batch_size', type=int,
                        default=512, help='Batch size')
    parser.add_argument('--num_pre_prune_epochs', type=int, default=10,
                        help='Number of epochs for pretraining')
    parser.add_argument('--num_prune_iterations', type=int, default=10,
                        help='Number of iterations to prune')
    parser.add_argument('--num_prune_epochs', type=int, default=10,
                        help='Number of epochs for pruning')
    parser.add_argument('--num_post_prune_epochs', type=int, default=100,
                        help='Number of epochs to run post training')
    parser.add_argument('--optimizer', type=str,
                        default='sgd', help='Optimizer', choices=['sgd'])
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.0, help='Momentum')
    parser.add_argument(
        '--post_prune_optimizer', type=str, default='adam',
        help='Post Prune Optimizer', choices=['adam', 'sgd'])
    parser.add_argument(
        '--post_prune_lr', type=float, default=3e-4,
        help='Learning rate for the post prune optimizer')
    parser.add_argument(
        '--pruner', type=str, default='causalpruner',
        help='Method for pruning', choices=['causalpruner', 'magpruner'])
    parser.add_argument(
        '--checkpoint_dir', type=str, default='../checkpoints',
        help='Checkpoint dir to write model weights and losses')
    parser.add_argument(
        '--start_clean', type=bool, default=True,
        help='Controls if the pruner deletes any existing directories when starting')
    parser.add_argument(
        '--causal_pruner_init_lr', type=float, default=1e-2,
        help='Learning rate for causal pruner')
    parser.add_argument(
        '--causal_pruner_l1_regularization_coeff', type=float, default=1e-7,
        help='Causal Pruner L1 regularization coefficient')
    parser.add_argument(
        '--causal_pruner_max_iter', type=int, default=1000,
        help='Maximum number of iterations to run causal pruner training')
    parser.add_argument(
        '--causal_pruner_loss_tol', type=float, default=1e-4,
        help='Loss tolerance between current loss and best loss for early stopping')
    parser.add_argument(
        '--causal_pruner_num_iter_no_change', type=int, default=5,
        help='Number of iterations with no loss improvement before declaring convergence')
    parser.add_argument(
        '--causal_pruner_backend', type=str, default='sklearn', 
        choices=['sklearn', 'torch'],
        help='Causal weights trainer backend')
    parser.add_argument('--mag_pruner_amount', type=float,
                        default=0.4, help='Magnitude pruning fraction')
    parser.add_argument(
        '--verbose', type=bool, default=True, help='Output verbosity')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
