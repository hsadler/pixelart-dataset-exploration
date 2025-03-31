from typing import Callable
from enum import Enum

import torch


class OptimizerType(Enum):
    ADAM = "adam"
    SGD = "sgd"
    RMSprop = "rmsprop"
    ADAMW = "adamw"


optimizer_map: dict[OptimizerType, Callable[[torch.nn.Module, float], torch.optim.Optimizer]] = {
    OptimizerType.ADAM: torch.optim.Adam,
    OptimizerType.SGD: torch.optim.SGD,
    OptimizerType.RMSprop: torch.optim.RMSprop,
    OptimizerType.ADAMW: torch.optim.AdamW,
}


def get_optimizer(
    optimizer_type: OptimizerType, model: torch.nn.Module, learning_rate: float
) -> torch.optim.Optimizer:
    try:
        return optimizer_map[optimizer_type](model.parameters(), lr=learning_rate)
    except KeyError:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
