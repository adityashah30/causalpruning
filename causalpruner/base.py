from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import shutil

from lightning.fabric import Fabric
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


@dataclass
class PrunerConfig:
    pruner: str
    fabric: Fabric
    model: nn.Module
    checkpoint_dir: str
    start_clean: bool
    reset_weights: bool
    reset_params: bool
    verbose: bool


class Pruner(ABC):
    _SUPPORTED_MODULES = [
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
    ]

    _MODULES_TO_RESET = [
        nn.BatchNorm1d,
        nn.BatchNorm2d,
    ]

    _PREFIXES_TO_CONSUME = ["_forward_module."]

    @staticmethod
    def is_module_supported(module: nn.Module) -> bool:
        for supported_module in Pruner._SUPPORTED_MODULES:
            if isinstance(module, supported_module):
                return True
        return False

    @staticmethod
    def is_module_to_be_reset(module: nn.Module) -> bool:
        for module_to_be_reset in Pruner._MODULES_TO_RESET:
            if isinstance(module, module_to_be_reset):
                return True
        return False

    @staticmethod
    def apply_identity_masks_to_model(model: nn.Module) -> None:
        for module in model.modules():
            if not Pruner.is_module_supported(module):
                continue
            prune.identity(module, "weight")

    def __init__(self, config: PrunerConfig):
        super().__init__()
        self.config = config
        self.fabric = config.fabric
        self.device = self.fabric.device
        self.iteration = -1

        self.modules_dict = dict()
        self.modules_to_reset = dict()
        for name, module in self.config.model.named_modules():
            for prefix in self._PREFIXES_TO_CONSUME:
                if name.startswith(prefix):
                    name = name.removeprefix(prefix)
            if self.is_module_supported(module):
                self.modules_dict[name] = module
            if self.is_module_to_be_reset(module):
                self.modules_to_reset[name] = module

        self.params = []
        for module_name, module in self.modules_dict.items():
            if hasattr(module, "weight"):
                self.params.append(module_name)
        self.params = sorted(self.params)

        self.checkpoint_dir = config.checkpoint_dir
        self.init_model_path = os.path.join(self.checkpoint_dir, "init.ckpt")
        self.loss_checkpoint_dir = os.path.join(self.checkpoint_dir, "loss")
        self.weights_checkpoint_dir = os.path.join(self.checkpoint_dir, "weights")

        # Setup directories on the global_rank = 0
        if self.fabric.is_global_zero:
            if config.start_clean and os.path.exists(self.checkpoint_dir):
                shutil.rmtree(self.checkpoint_dir)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            os.makedirs(self.loss_checkpoint_dir, exist_ok=True)
            os.makedirs(self.weights_checkpoint_dir, exist_ok=True)
        self.fabric.barrier()

    def __str__(self) -> str:
        return self.config.pruner

    def __repr__(self) -> str:
        return str(self)

    @abstractmethod
    def compute_masks(self) -> None:
        raise NotImplementedError(
            "Pruner is an abstract class. Use an appropriate derived class."
        )

    @torch.no_grad()
    def apply_masks(self) -> None:
        for param in self.params:
            module = self.modules_dict[param]
            prune.remove(module, "weight")

    @torch.no_grad()
    def apply_identity_masks(self) -> None:
        for param in self.params:
            module = self.modules_dict[param]
            prune.identity(module, "weight")

    @torch.no_grad()
    def remove_masks(self) -> None:
        for _, module in self.modules_dict.items():
            setattr(module, "weight", module.weight_orig)
            delattr(module, "weight_orig")
            delattr(module, "weight_mask")
            for k, hook in module._forward_pre_hooks.items():
                if isinstance(hook, prune.BasePruningMethod):
                    del module._forward_pre_hooks[k]

    @torch.no_grad()
    def provide_loss_before_step(self, loss: torch.tensor) -> None:
        pass

    @torch.no_grad()
    def provide_loss_after_step(self, loss: torch.tensor) -> None:
        pass

    @torch.no_grad()
    def start_pruning(self) -> None:
        if self.fabric.is_global_zero:
            self.fabric.save(self.init_model_path, {"model": self.config.model})
        self.fabric.barrier()

    @torch.no_grad()
    def start_iteration(self) -> None:
        self.iteration += 1
        self.counter = 0
        if self.fabric.is_global_zero:
            iteration_name = f"{self.iteration}"
            loss_dir = os.path.join(self.loss_checkpoint_dir, iteration_name)
            os.makedirs(loss_dir, exist_ok=True)
            weights_dir = os.path.join(self.weights_checkpoint_dir, iteration_name)
            os.makedirs(weights_dir, exist_ok=True)
        self.fabric.barrier()

    @torch.no_grad()
    def reset_params(self) -> None:
        if not self.config.reset_params:
            return
        for module in self.modules_to_reset.values():
            module.reset_parameters()

    @torch.no_grad()
    def reset_weights(self) -> None:
        self.config.model.zero_grad(set_to_none=True)
        if not self.config.reset_weights:
            return
        masks = dict()
        for name, module in self.modules_dict.items():
            if hasattr(module, "weight_mask"):
                masks[name] = getattr(module, "weight_mask")
        self.remove_masks()
        self.fabric.load(self.init_model_path, {"model": self.config.model})
        for name, module in self.modules_dict.items():
            if name in masks:
                prune.custom_from_mask(module, "weight", masks[name])
