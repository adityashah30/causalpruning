from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CausalWeightsTrainer(nn.Module):

    def __init__(self, model_weights: torch.Tensor):
        super().__init__()
        flattened_dims = np.prod(model_weights.size(), dtype=int)
        self.layer = nn.Linear(flattened_dims, 1, bias=False)
        self.optimizer = optim.Adam(self.layer.parameters())

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layer(X)

    def fit(self, X: torch.Tensor, Y: torch.Tensor):
        self.layer.train()
        Y_hat = torch.squeeze(self.forward(X), dim=1)
        Y = torch.flatten(Y)
        loss = F.mse_loss(Y_hat, Y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_weights(self) -> torch.Tensor:
        return torch.flatten(self.layer.weight.detach().clone())


class CausalPruner(ABC, nn.Module):

    _SUPPORTED_MODULES = [
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
    ]

    @staticmethod
    def is_module_supported(module: nn.Module) -> bool:
        for supported_module in CausalPruner._SUPPORTED_MODULES:
            if isinstance(module, supported_module):
                return True
        return False

    def __init__(self, model: nn.Module):
        super().__init__()

        self.modules_dict = nn.ModuleDict()
        for name, module in model.named_children():
            if self.is_module_supported(module):
                self.modules_dict[name] = module

        self.params_dict = nn.ParameterDict()
        self.causal_weights_trainers = nn.ModuleDict()
        for module_name, module in self.modules_dict.items():
            for param_name, param in module.named_parameters():
                if 'weight' not in param_name:
                    continue
                device = param.device
                self.params_dict[module_name] = param
                self.causal_weights_trainers[module_name] = CausalWeightsTrainer(
                    param).to(device=device)

    @abstractmethod
    def provide_loss(self, loss: torch.Tensor) -> None:
        pass

    @abstractmethod
    def compute_masks(self) -> None:
        pass
