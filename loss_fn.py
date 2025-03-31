from typing import Callable
from enum import Enum

import torch


class LossType(Enum):
    MSE = "mse"
    L1 = "l1"
    HYBRID = "hybrid"  # Combined MSE and L1
    SSIM = "ssim"


def _mse_loss_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.nn.MSELoss()(x, y)


def _l1_loss_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.nn.L1Loss()(x, y)


def _hybrid_loss_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.nn.MSELoss()(x, y) + 0.5 * torch.nn.L1Loss()(x, y)


def _ssim_loss_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 1 - torch.mean(
        (x * y).sum(dim=(1, 2, 3))
        / (torch.sqrt((x * x).sum(dim=(1, 2, 3))) * torch.sqrt((y * y).sum(dim=(1, 2, 3))))
    )


loss_fn_map: dict[LossType, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    LossType.MSE: _mse_loss_fn,
    LossType.L1: _l1_loss_fn,
    LossType.HYBRID: _hybrid_loss_fn,
    LossType.SSIM: _ssim_loss_fn,
}


def get_loss_function(loss_type: LossType) -> torch.nn.Module:
    try:
        return loss_fn_map[loss_type]
    except KeyError:
        raise ValueError(f"Unknown loss type: {loss_type}")
