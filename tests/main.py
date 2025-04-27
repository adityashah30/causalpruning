# autopep8: off
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# autopep: on

import argparse
import shutil
from typing import Optional

from lightning.fabric import Fabric
import torch
import torch.nn as nn
import torch.optim as optim

from causalpruner import (
    CausalWeightsTrainerConfig,
    Pruner,
    PrunerConfig,
    SGDPruner,
    SGDPrunerConfig,
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
    LRRangeFinderConfig,
    Pruner,
    Trainer,
    TrainerConfig,
)


torch.set_float32_matmul_precision("medium")


def delete_dir_if_exists(dir: str):
    if os.path.exists(dir):
        shutil.rmtree(dir)


def get_prune_optimizer(name: str, model: nn.Module, lr: float) -> optim.Optimizer:
    name = name.lower()
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr)
    raise NotImplementedError(f"{name} is not a supported Optimizier")


def get_train_optimizer(name: str, model: nn.Module, lr: float) -> optim.Optimizer:
    name = name.lower()
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif name == "sgd":
        return optim.SGD(model.parameters(), lr=lr)
    elif name == "sgd_momentum":
        return optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4
        )
    raise NotImplementedError(f"{name} is not a supported post-prune Optimizier")


def get_pruner(pruner_config: PrunerConfig) -> Pruner:
    if isinstance(pruner_config, SGDPrunerConfig):
        return SGDPruner(pruner_config)
    elif isinstance(pruner_config, MagPrunerConfig):
        return MagPruner(pruner_config)
    raise NotImplementedError(f"{type(pruner_config)} is not supported yet")


def main(args):
    if args.verbose:
        print(args)
    prune = args.prune
    pruner = args.pruner
    model_name = args.model
    dataset_name = args.dataset
    prune_identifier = pruner if prune else "noprune"
    if prune_identifier == "causalpruner":
        iteration_id = (
            f"{args.num_prune_iterations}_"
            + f"{args.num_train_epochs_before_pruning}_"
            + f"{args.num_prune_epochs}"
        )
        alpha_id = f"{args.causal_pruner_l1_regularization_coeff}"
        prune_identifier += f"_{iteration_id}_{alpha_id}"
    elif prune_identifier == "magpruner":
        prune_identifier += f"_{args.mag_pruner_amount}"
    identifier = f"{model_name}_{dataset_name}_{prune_identifier}"
    suffix = args.suffix
    if suffix != "":
        identifier = f"{identifier}_{suffix}"
    checkpoint_dir = os.path.join(args.checkpoint_dir, identifier)
    tensorboard_dir = os.path.join(args.tensorboard_dir, identifier)
    if args.train_only:
        args.start_clean = False
    if args.start_clean:
        delete_dir_if_exists(checkpoint_dir)
        delete_dir_if_exists(tensorboard_dir)
    train_dataset, test_dataset, num_classes = get_dataset(
        dataset_name,
        model_name,
        args.dataset_root_dir,
        cache_size_limit_gb=args.dataset_cache_size_limit_gb,
    )
    fabric = Fabric(devices=args.device_ids, accelerator="auto")
    fabric.launch()
    model = get_model(model_name, dataset_name)
    prune_optimizer = get_prune_optimizer(args.optimizer, model, args.lr)
    train_optimizer = get_train_optimizer(args.train_optimizer, model, args.train_lr)
    model, prune_optimizer, train_optimizer = fabric.setup(
        model, prune_optimizer, train_optimizer
    )
    world_size = fabric.world_size
    batch_size = args.batch_size // world_size
    data_config = DataConfig(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=args.num_dataset_workers,
        pin_memory=args.pin_memory,
        shuffle=args.shuffle_dataset,
        num_classes=num_classes,
    )
    epoch_config = EpochConfig(
        num_pre_prune_epochs=args.num_pre_prune_epochs if args.prune else 0,
        num_prune_iterations=args.num_prune_iterations if args.prune else 0,
        num_train_epochs_before_pruning=args.num_train_epochs_before_pruning
        if args.prune
        else 0,
        num_prune_epochs=args.num_prune_epochs if args.prune else 0,
        num_train_epochs=args.max_train_epochs,
        num_batches_in_epoch=args.num_batches_in_epoch,
        tqdm_update_frequency=args.tqdm_update_frequency,
    )
    lrrt_config = LRRangeFinderConfig(
        enable=args.run_lrrt,
        min_lr=args.lrrt_min_lr,
        max_lr=args.lrrt_max_lr,
        num_steps=args.lrrt_num_steps,
        ewa_alpha=args.lrrt_ewa_alpha,
    )
    trainer_config = TrainerConfig(
        hparams=vars(args),
        fabric=fabric,
        model=model,
        prune_optimizer=prune_optimizer,
        train_optimizer=train_optimizer,
        lrrt_config=lrrt_config,
        train_convergence_loss_tolerance=args.train_convergence_loss_tolerance,
        train_loss_num_epochs_no_change=args.train_loss_num_epochs_no_change,
        data_config=data_config,
        epoch_config=epoch_config,
        tensorboard_dir=tensorboard_dir,
        checkpoint_dir=checkpoint_dir,
        verbose=args.verbose,
        train_only=args.train_only,
        model_to_load_for_training=args.model_to_load_for_training,
        model_to_save_after_training=args.model_to_save_after_training,
        use_one_cycle_lr_scheduler=args.use_one_cycle_lr_scheduler,
    )
    pruner = None
    if args.prune:
        total_prune_amount = args.total_prune_amount
        num_prune_iterations = args.num_prune_iterations
        prune_amount_per_iteration = 1 - (1 - total_prune_amount) ** (
            1 / num_prune_iterations
        )
        print(f"Prune amount per iteration: {prune_amount_per_iteration}")
        if args.pruner == "causalpruner":
            causal_weights_trainer_config = CausalWeightsTrainerConfig(
                fabric=fabric,
                init_lr=args.causal_pruner_init_lr,
                l1_regularization_coeff=args.causal_pruner_l1_regularization_coeff,
                prune_amount=prune_amount_per_iteration,
                max_iter=args.causal_pruner_max_iter,
                loss_tol=args.causal_pruner_loss_tol,
                num_iter_no_change=args.causal_pruner_num_iter_no_change,
                backend=args.causal_pruner_backend,
            )
            pruner_config = SGDPrunerConfig(
                fabric=fabric,
                model=model,
                pruner="SGDPruner",
                checkpoint_dir=checkpoint_dir,
                start_clean=args.start_clean,
                eval_after_epoch=args.eval_after_epoch,
                reset_weights=args.reset_weights_after_pruning,
                batch_size=args.causal_pruner_batch_size,
                num_dataloader_workers=args.num_causal_pruner_dataloader_workers,
                pin_memory=args.causal_pruner_pin_memory,
                threaded_checkpoint_writer=args.causal_pruner_threaded_checkpoint_writer,
                delete_checkpoint_dir_after_training=args.delete_checkpoint_dir_after_training,
                trainer_config=causal_weights_trainer_config,
            )
        elif args.pruner == "magpruner":
            pruner_config = MagPrunerConfig(
                fabric=fabric,
                model=model,
                pruner="MagPruner",
                checkpoint_dir=checkpoint_dir,
                start_clean=args.start_clean,
                eval_after_epoch=args.eval_after_epoch,
                reset_weights=args.reset_weights_after_pruning,
                prune_amount=args.mag_pruner_amount,
            )
        pruner = get_pruner(pruner_config)
    trainer = Trainer(trainer_config, pruner)
    trainer.run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Causal Pruning")

    parser.add_argument(
        "--device_ids",
        type=str,
        default="-1",
        help="The device id. Useful for multi device systems",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix to be used for identifier. Not used if empty -- else adds `_{suffix}` to the identifier.",
    )
    # Model args
    parser.add_argument(
        "--model",
        type=str,
        choices=["alexnet", "lenet", "resnet18", "resnet50", "resnet50_untrained"],
        default="lenet",
        help="Model name",
    )
    parser.add_argument(
        "--train_convergence_loss_tolerance",
        type=float,
        default=1e-4,
        help="Considers the model converged when train loss does not change by more than this value for train_loss_num_epochs_no_change",
    )
    parser.add_argument(
        "--train_loss_num_epochs_no_change",
        type=int,
        default=5,
        help="Considers the model converged when train loss does not change by more than train_convergence_loss_tolerance for these many epochs",
    )
    parser.add_argument(
        "--max_train_epochs",
        type=int,
        default=200,
        help="Maximum number of epochs for train the model",
    )
    parser.add_argument(
        "--train_optimizer",
        type=str,
        default="sgd_momentum",
        help="Training Optimizer",
        choices=["adam", "sgd", "sgd_momentum"],
    )
    parser.add_argument(
        "--train_lr",
        type=float,
        default=0.1,
        help="Learning rate for the train optimizer",
    )
    parser.add_argument(
        "--use_one_cycle_lr_scheduler",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Uses the one cycle lr scheduler when train_optimzier is either `sgd` or `sgd_momentum`",
    )
    parser.add_argument(
        "--run_lrrt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Runs the Learning Rate Range Finder test if enabled",
    )
    parser.add_argument(
        "--lrrt_max_lr", type=float, default=10.0, help="Max LR to use for LRRT"
    )
    parser.add_argument(
        "--lrrt_min_lr", type=float, default=1e-7, help="Min LR to use for LRRT"
    )
    parser.add_argument(
        "--lrrt_num_steps", type=int, default=1000, help="Number of steps to run LRRT"
    )
    parser.add_argument(
        "--lrrt_ewa_alpha",
        type=float,
        default=0.98,
        help="Smoothing factor used for LRRT",
    )
    parser.add_argument(
        "--train_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only trains the model if set to true. i.e. doesn't do pre-pruning or pruning",
    )
    parser.add_argument(
        "--model_to_load_for_training",
        type=str,
        default="prune.final",
        help='Model id to load for training. The loaded path is "model.{model_to_load_for_training}.ckpt',
    )
    parser.add_argument(
        "--model_to_save_after_training",
        type=str,
        default="trained",
        help='Model id to save post training. The saved model path is "model.{model_to_save_after_training}.ckpt',
    )
    # Dataset args
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "fashionmnist", "imagenet", "tinyimagenet"],
        default="cifar10",
        help="Dataset name",
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default="../data",
        help="Directory to download datasets",
    )
    parser.add_argument(
        "--num_dataset_workers", type=int, default=6, help="Number of dataset workers"
    )
    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pins memory to GPU when enabled.",
    )
    parser.add_argument(
        "--shuffle_dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to shuffle the train and test datasets",
    )
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument(
        "--dataset_cache_size_limit_gb",
        type=int,
        default=16,
        help="Size limit for dataset stochastic cache",
    )
    # Dirs
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="../checkpoints",
        help="Checkpoint dir to write model weights and losses",
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default="../tensorboard",
        help="Directory to write tensorboard data",
    )
    # Pruner args
    parser.add_argument(
        "--prune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prunes the model and then trains it if set to true -- else just trains it until convergence",
    )
    parser.add_argument(
        "--num_pre_prune_epochs",
        type=int,
        default=10,
        help="Number of epochs for pretraining",
    )
    parser.add_argument(
        "--num_prune_iterations",
        type=int,
        default=10,
        help="Number of iterations to prune",
    )
    parser.add_argument(
        "--num_train_epochs_before_pruning",
        type=int,
        default=0,
        help="Number of epochs for training before pruning in each pruning iteration",
    )
    parser.add_argument(
        "--num_prune_epochs", type=int, default=10, help="Number of epochs for pruning"
    )
    parser.add_argument(
        "--num_batches_in_epoch",
        type=int,
        default=-1,
        help="Number of batches per epoch. Runs the entire epoch by default",
    )
    parser.add_argument(
        "--tqdm_update_frequency", type=int, default=1, help="tqdm update frequency."
    )
    parser.add_argument(
        "--optimizer", type=str, default="sgd", help="Optimizer", choices=["sgd"]
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--eval_after_epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Eval after each pruning epoch",
    )
    parser.add_argument(
        "--pruner",
        type=str,
        default="causalpruner",
        help="Method for pruning",
        choices=["causalpruner", "magpruner"],
    )
    parser.add_argument(
        "--reset_weights_after_pruning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset weights to an earlier checkpoint after prune step if true.",
    )
    parser.add_argument(
        "--causal_pruner_threaded_checkpoint_writer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Controls if weights are written using a ThreadPoolExecutor",
    )
    parser.add_argument(
        "--start_clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Controls if the pruner deletes any existing directories when starting",
    )
    parser.add_argument(
        "--causal_pruner_init_lr",
        type=float,
        default=1e-2,
        help="Learning rate for causal pruner",
    )
    parser.add_argument(
        "--causal_pruner_l1_regularization_coeff",
        type=float,
        default=1e-3,
        help="Causal Pruner L1 regularization coefficient",
    )
    parser.add_argument(
        "--causal_pruner_max_iter",
        type=int,
        default=1000,
        help="Maximum number of iterations to run causal pruner training",
    )
    parser.add_argument(
        "--causal_pruner_loss_tol",
        type=float,
        default=1e-4,
        help="Loss tolerance between current loss and best loss for early stopping",
    )
    parser.add_argument(
        "--causal_pruner_num_iter_no_change",
        type=int,
        default=2,
        help="Number of iterations with no loss improvement before declaring convergence",
    )
    parser.add_argument(
        "--causal_pruner_batch_size",
        type=int,
        default=64,
        help="Batch size for causal pruner training. Use -1 to use the entire dataset",
    )
    parser.add_argument(
        "--num_causal_pruner_dataloader_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers to use while training prune weights",
    )
    parser.add_argument(
        "--causal_pruner_pin_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Controls if the causal pruner dataloader uses pinned memory.",
    )
    parser.add_argument(
        "--delete_checkpoint_dir_after_training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deletes the checkpoint directory once params are trained. Used to save space.",
    )
    parser.add_argument(
        "--causal_pruner_backend",
        type=str,
        default="torch",
        choices=["sklearn", "torch"],
        help="Causal weights trainer backend",
    )
    parser.add_argument(
        "--total_prune_amount",
        type=float,
        default=0.9,
        help="Total prune anount after all the iterations",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Output verbosity",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
