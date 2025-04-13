# Copyright (c) RenChu Wang - All Rights Reserved

from torch import Tensor
from torch.nn import functional as F

from .dists import DistLoss


class BceDistLoss(DistLoss):
    def _compute(self, input, target):
        return F.binary_cross_entropy(input=input, target=target)


class L1DistLoss(DistLoss):
    def _compute(self, input: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(input=input, target=target)


class MseDistLoss(DistLoss):
    def _compute(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input=input, target=target)


class KlDivDistLoss(DistLoss):
    def _compute(self, input: Tensor, target: Tensor) -> Tensor:
        return F.kl_div(input=input, target=target)


class SmoothL1DistLoss(DistLoss):
    def _compute(self, input: Tensor, target: Tensor) -> Tensor:
        return F.smooth_l1_loss(input=input, target=target)


class CrossEntropyDistLoss(DistLoss):
    def _compute(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(input=input, target=target)


class NllDistLoss(DistLoss):
    def _compute(self, input: Tensor, target: Tensor) -> Tensor:
        return F.nll_loss(input=input, target=target)
