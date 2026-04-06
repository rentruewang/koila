# Copyright (c) AIoWay Authors - All Rights Reserved

"The loss modules."

import typing
from collections import abc as cabc

import torch
from torch import nn

from aioway._common import dcls_no_eq_no_repr

from .tensors import TensorFn

__all__ = ["LossFn"]


@dcls_no_eq_no_repr
class LossFn[**P, M: nn.Module](TensorFn):
    """
    `ModuleFn` informs us how `nn.Module` would behave without initializing it.
    """

    input: TensorFn
    target: TensorFn
    loss: cabc.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    def __post_init__(self):
        # Must output a scalar.
        if self.preview().numel() != 1:
            raise AssertionError("The loss function does not output a scalar.")

    @typing.override
    def forward(self) -> torch.Tensor:
        input = self.input.do()
        target = self.target.do()
        return self.loss(input, target)
