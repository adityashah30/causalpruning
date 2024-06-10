import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# autopep8: off
import causalpruner
from causalpruner import PrunerConfig
from causalpruner import SGDPrunerConfig, SGDPrunerTrainer
from causalpruner import Trainer, TrainerConfig
from pruner.mag_pruner import MagPrunerConfig, MagPrunerTrainer
#autopep on


def get_trainer(
        pruner: str, config: TrainerConfig,
        pruner_config: PrunerConfig) -> Trainer:
    pruner = pruner.lower()
    if pruner == 'causalpruner' and isinstance(pruner_config, SGDPrunerConfig):
        return SGDPrunerTrainer(config, pruner_config)
    elif pruner == 'magpruner' and isinstance(pruner_config, MagPrunerConfig):
        return MagPrunerTrainer(config, pruner_config)
    raise NotImplementedError('Please use a valid pruner')
