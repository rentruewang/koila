# Copyright (c) AIoWay Authors - All Rights Reserved

from torch import Tensor
from torch.nn import functional as F

from .losses import LossFunc

__all__ = [
    "BceDistLoss",
    "L1DistLoss",
    "MseDistLoss",
    "KlDivDistLoss",
    "SmoothL1DistLoss",
    "CrossEntropyDistLoss",
    "NllDistLoss",
]


class BceDistLoss(LossFunc):
    def _compute(self, input, target):
        return F.binary_cross_entropy(input=input, target=target)


class L1DistLoss(LossFunc):
    def _compute(self, input: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(input=input, target=target)


class MseDistLoss(LossFunc):
    def _compute(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input=input, target=target)


class KlDivDistLoss(LossFunc):
    def _compute(self, input: Tensor, target: Tensor) -> Tensor:
        return F.kl_div(input=input, target=target)


class SmoothL1DistLoss(LossFunc):
    def _compute(self, input: Tensor, target: Tensor) -> Tensor:
        return F.smooth_l1_loss(input=input, target=target)


class CrossEntropyDistLoss(LossFunc):
    def _compute(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(input=input, target=target)


class NllDistLoss(LossFunc):
    def _compute(self, input: Tensor, target: Tensor) -> Tensor:
        return F.nll_loss(input=input, target=target)
