from dataclasses import dataclass
from causalpruner import PrunerConfig, Pruner, SGDPruner
from pruner.mag_pruner import MagPruner
@dataclass 
class CombinedPrunerConfig(PrunerConfig):
    phase_iterations: dict[str, int]  
    order: list[str]
    pruner_configs: dict[str, PrunerConfig] 

class CombinedPruner(Pruner):
    def __init__(self, config: CombinedPrunerConfig):
        super().__init__(config)
        self.phase_index = 0
        self.current_phase_iteration = 0
        self.active_pruners = {} 
        
        
        self.available_pruners = {
            "causal": lambda: SGDPruner(config.pruner_configs["causal"]),
            "mag": lambda: MagPruner(config.pruner_configs["mag"])
        }
        
        
        current_phase = config.order[0]
        self._init_phase(current_phase)

    def _init_phase(self, phase: str):
        if phase not in self.active_pruners:
            pruner = self.available_pruners[phase]()

            pruner.start_pruning()
            total_iters = self.config.phase_iterations[phase]
            self.active_pruners[phase] = (pruner, total_iters)

    def start_pruning(self) -> None:
        pruner, _ = self._current_phase()
        pruner.start_pruning()

    def _current_phase(self) -> tuple[Pruner, int]:
        phase_name = self.config.order[self.phase_index]
        pruner, total_iters = self.active_pruners[phase_name]
        return pruner, total_iters
    

    def start_iteration(self):
        pruner, _ = self._current_phase()
        pruner.start_iteration()

    def compute_masks(self):
        pruner, _ = self._current_phase()
        pruner.compute_masks()

    def provide_loss_before_step(self, loss):
        pruner, _ = self._current_phase()
        pruner.provide_loss_before_step(loss)

    def provide_loss_after_step(self, loss):
        pruner, total_iters = self._current_phase()
        pruner.provide_loss_after_step(loss)

    def next_phase(self):
        new_phase = self.config.order[1]
        self.phase_index = 1
        self._init_phase(new_phase)

    def reset_weights(self):
        pruner, _ = self._current_phase()

        if any(hasattr(module, "weight_orig") for module in pruner.modules_dict.values()):
            pruner.reset_weights()
    
    def compute_masks(self):
        pruner, _ = self._current_phase()
        pruner.compute_masks()