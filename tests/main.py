# autopep8: off
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# autopep: on

import argparse
import shutil

from lightning.fabric import Fabric
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    OneCycleLR,
)
from torchvision.transforms import v2

from causalpruner import (
    CausalWeightsTrainerConfig,
    Pruner,
    PrunerConfig,
    SGDPruner,
    SGDPrunerConfig,
)
from datasets import get_dataset
from models import get_model
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


torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True


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
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    raise NotImplementedError(f"{name} is not a supported post-prune Optimizier")


def _calculate_total_epochs(epoch_config: EpochConfig) -> int:
    num_epochs = (
        epoch_config.num_pre_prune_epochs
        + (
            (epoch_config.num_prune_iterations - 1)
            * epoch_config.num_train_epochs_before_pruning
        )
        + epoch_config.num_train_epochs
    )
    return num_epochs


def _calculate_total_steps(epoch_config: EpochConfig, data_config: DataConfig) -> int:
    num_items = len(data_config.train_dataset)
    batch_size = data_config.batch_size
    num_epochs = _calculate_total_epochs(epoch_config)
    num_batches_in_epoch = epoch_config.num_batches_in_epoch

    num_batches = (num_items + batch_size - 1) // batch_size
    if num_batches_in_epoch <= 0:
        num_batches_in_epoch = num_batches

    total_steps = num_epochs * num_batches_in_epoch
    return total_steps


def create_lr_scheduler(
    lr_scheduler: str,
    optimizer: optim.Optimizer,
    epoch_config: EpochConfig,
    data_config: DataConfig,
    train_lr: float,
    max_train_lr: float,
) -> LRScheduler:
    lr_scheduler = lr_scheduler.lower()
    if lr_scheduler == "onecycle":
        total_steps = _calculate_total_steps(epoch_config, data_config)
        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=max_train_lr,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=max_train_lr / train_lr,
            final_div_factor=10**4,
        )
        return lr_scheduler
    elif lr_scheduler == "cosineannealing":
        total_steps = _calculate_total_epochs(epoch_config)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
        return lr_scheduler
    return None


def get_pruner(pruner_config: PrunerConfig) -> Pruner:
    if isinstance(pruner_config, SGDPrunerConfig):
        return SGDPruner(pruner_config)
    elif isinstance(pruner_config, MagPrunerConfig):
        return MagPruner(pruner_config)
    raise NotImplementedError(f"{type(pruner_config)} is not supported yet")


def get_collate_fn(mixup_alpha: float, cutmix_alpha: float, num_classes: int):
    transforms = []

    if mixup_alpha > 0:
        transforms.append(v2.MixUp(alpha=mixup_alpha, num_classes=num_classes))
    if cutmix_alpha > 0:
        transforms.append(v2.CutMix(alpha=cutmix_alpha, num_classes=num_classes))

    if len(transforms) == 0:
        return None

    return v2.RandomChoice(transforms)


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
        prune_amount = f"{args.total_prune_amount}"
        prune_identifier += f"_{iteration_id}_{alpha_id}_{prune_amount}"
    elif prune_identifier == "magpruner":
        prune_identifier += f"_{args.total_prune_amount}"
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
    )
    collate_fn = get_collate_fn(
        args.mixup_alpha, args.cutmix_alpha, num_classes=num_classes
    )
    fabric = Fabric(
        devices=args.device_ids, accelerator="auto", precision=args.precision
    )
    fabric.launch()
    model = get_model(model_name, dataset_name)
    if args.compile_model:
        model = torch.compile(model)
    prune_optimizer = get_prune_optimizer(args.optimizer, model, args.lr)
    train_optimizer = get_train_optimizer(args.train_optimizer, model, args.train_lr)
    model, prune_optimizer, train_optimizer = fabric.setup(
        model, prune_optimizer, train_optimizer
    )
    world_size = fabric.world_size
    batch_size = args.batch_size // world_size
    batch_size_while_pruning = args.batch_size_while_pruning // world_size
    data_config = DataConfig(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        batch_size_while_pruning=batch_size_while_pruning,
        num_workers=args.num_dataset_workers,
        pin_memory=args.pin_memory,
        shuffle=args.shuffle_dataset,
        num_classes=num_classes,
        collate_fn=collate_fn,
    )
    epoch_config = EpochConfig(
        num_pre_prune_epochs=args.num_pre_prune_epochs if args.prune else 0,
        num_prune_iterations=args.num_prune_iterations if args.prune else 0,
        num_train_epochs_before_pruning=args.num_train_epochs_before_pruning
        if args.prune
        else 0,
        num_prune_epochs=args.num_prune_epochs if args.prune else 0,
        num_train_epochs=args.num_train_epochs,
        num_batches_in_epoch=args.num_batches_in_epoch,
        tqdm_update_frequency=args.tqdm_update_frequency,
    )
    lr_scheduler = None
    if args.lr_scheduler != "":
        lr_scheduler = create_lr_scheduler(
            args.lr_scheduler,
            train_optimizer,
            epoch_config,
            data_config,
            args.train_lr,
            args.max_train_lr,
        )
    trainer_config = TrainerConfig(
        hparams=vars(args),
        fabric=fabric,
        model=model,
        prune_optimizer=prune_optimizer,
        train_optimizer=train_optimizer,
        data_config=data_config,
        epoch_config=epoch_config,
        tensorboard_dir=tensorboard_dir,
        checkpoint_dir=checkpoint_dir,
        lr_scheduler=lr_scheduler,
        verbose=args.verbose,
        train_only=args.train_only,
        model_to_load_for_training=args.model_to_load_for_training,
        model_to_save_after_training=args.model_to_save_after_training,
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
                initialization=args.causal_pruner_weights_initialization,
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
                verbose=args.verbose,
                num_dataloader_workers=args.num_causal_pruner_dataloader_workers,
                pin_memory=args.causal_pruner_pin_memory,
                threaded_checkpoint_writer=args.causal_pruner_threaded_checkpoint_writer,
                delete_checkpoint_dir_after_training=args.delete_checkpoint_dir_after_training,
                use_zscaling=args.causal_pruner_use_zscaling,
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
                prune_amount=prune_amount_per_iteration,
                verbose=args.verbose,
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
        "--precision",
        type=str,
        default="32",
        choices=[
            "32",
            "16-mixed",
            "bf16-mixed",
            "16-true",
            "bf16-true",
            "transformer-engine",
        ],
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
        choices=[
            "alexnet",
            "lenet",
            "mlpnet",
            "resnet18",
            "resnet20",
            "resnet50",
            "resnet50_untrained",
        ],
        default="lenet",
        help="Model name",
    )
    parser.add_argument(
        "--compile_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile the model for faster execution.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=300,
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
        default=1e-3,
        help="Learning rate for the train optimizer",
    )
    parser.add_argument(
        "--max_train_lr",
        type=float,
        default=0.1,
        help="Maximum training learning rate to use with OneCycleLR",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="onecycle",
        choices=["", "onecycle", "cosineannealing"],
        help="Use the LR scheduler when training",
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
        choices=["cifar10", "fashionmnist", "imagenet", "mnist", "tinyimagenet"],
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
        "--num_dataset_workers", type=int, default=8, help="Number of dataset workers"
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
    parser.add_argument(
        "--mixup_alpha",
        type=float,
        default=-1,
        help="Mixup alpha for MixUp augmentations",
    )
    parser.add_argument(
        "--cutmix_alpha",
        type=float,
        default=-1,
        help="Mixup alpha for CutMix augmentations",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
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
        default=10,
        help="Number of epochs for training before pruning in each pruning iteration",
    )
    parser.add_argument(
        "--num_prune_epochs", type=int, default=1, help="Number of epochs for pruning"
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
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Prune optimizer learning rate"
    )
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
        default=False,
        help="Reset weights to an earlier checkpoint after prune step if true.",
    )
    parser.add_argument(
        "--batch_size_while_pruning",
        type=int,
        default=256,
        help="Batch size while pruning",
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
        default=0,
        help="Causal Pruner L1 regularization coefficient",
    )
    parser.add_argument(
        "--causal_pruner_weights_initialization",
        type=str,
        default="zeros",
        choices=["zeros", "xavier_normal"],
        help="Causal Pruner model weights initialization scheme",
    )
    parser.add_argument(
        "--causal_pruner_max_iter",
        type=int,
        default=30,
        help="Maximum number of iterations to run causal pruner training",
    )
    parser.add_argument(
        "--causal_pruner_use_zscaling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Controls whether causal pruner model uses zscaling or not",
    )
    parser.add_argument(
        "--causal_pruner_loss_tol",
        type=float,
        default=1e-6,
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
        default=16,
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
