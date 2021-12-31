from __future__ import annotations

import logging
from dataclasses import dataclass
from numbers import Number
from typing import Tuple, overload

import torch
from numpy import ndarray
from rich.logging import RichHandler
from torch import Tensor

from .runnable_tensors import BatchInfo
from .runnables import Runnable
from .tensors import TensorLike

logger = logging.getLogger(__name__)
logger.addHandler(RichHandler())
logger.setLevel(logging.DEBUG)


class ImmediateTensor(Tensor, TensorLike):
    """
    Immediate tensor is a thin wrapper for the `Tensor` class. It's basically a tensor.
    """

    batch: BatchInfo | None = None

    def run(self, partial: Tuple[int, int] | None = None) -> Tensor:
        del partial

        return self


@dataclass
class ImmediateNumber(Runnable[Number]):
    data: Number

    def run(self, partial: Tuple[int, int] | None = None) -> Number:
        del partial

        return self.data


@overload
def wrap(tensor: Tensor | ndarray) -> ImmediateTensor:
    ...


@overload
def wrap(tensor: Number) -> ImmediateNumber:
    ...


def wrap(tensor: Tensor | ndarray | Number) -> ImmediateTensor | ImmediateNumber:
    if isinstance(tensor, ndarray):
        tensor = torch.from_numpy(tensor)

    if isinstance(tensor, Number):
        return ImmediateNumber(tensor)

    if isinstance(tensor, Tensor):
        return tensor.as_subclass(ImmediateTensor)  # type: ignore

    raise ValueError
