# Copyright (c) AIoWay Authors - All Rights Reserved

"The optimizer `Fn`."

import typing
from collections import abc as cabc

import torch
from torch import optim

from .fn import Fn

__all__ = ["OptimFn"]


class OptimType(typing.Protocol):
    def __call__(
        self, params: cabc.Iterable[torch.Tensor], lr: float
    ) -> optim.Optimizer: ...


class OptimFn[O: optim.Optimizer](Fn[None]):
    """
    The optimizer `Fn`, used to do the step operation.

    The init signature follows closly from `torch.optim.*` optimizers,
    having `lr` and `params` as input (to pass into `params`).
    """

    def __init__(
        self,
        optim_cls: OptimType,
        params: cabc.Iterable[torch.Tensor],
        lr: float,
    ) -> None:
        self._optimizer = optim_cls(params=params, lr=lr)

    @typing.override
    def forward(self) -> None:
        """
        Calls `.step()`, should do nothing in fake mode.
        """

        self.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def step(self) -> None:
        self._optimizer.step()
