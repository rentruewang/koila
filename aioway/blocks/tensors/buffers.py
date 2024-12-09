# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import operator
import typing
from collections.abc import Callable, Iterator
from types import FunctionType
from typing import Self

import numpy as np
import torch
from pandas import Series
from torch import Tensor

from aioway.blocks._typing import TensorNumber
from aioway.blocks.blocks import Buffer

from . import utils

__all__ = ["TensorBuffer"]


def _unary_op[S: "TensorBuffer", E](op: Callable[[Tensor], E]) -> FunctionType:
    def func(self: S) -> E:
        return op(self.data)

    assert isinstance(func, FunctionType)
    return func


def _binary_op[
    S: "TensorBuffer"
](op: Callable[[Tensor, TensorNumber], Tensor]) -> FunctionType:
    def func(self: S, other: S | TensorNumber) -> S:
        if not isinstance(other, TensorBuffer):
            other_data = other
        else:
            other_data = other.data

        return type(self)(data=op(self.data, other_data))

    assert isinstance(func, FunctionType)
    return func


@typing.final
@dcls.dataclass(frozen=True)
class TensorBuffer(Buffer):
    """
    ``Buffer`` represenets an in-memory column,
    acting as the most primitive operation in ``aioway``.

    It is a thin wrapper for `torch.Tensor`, while providing additional checks,
    as well as interfacing with the data structures in the project.
    It handles indexing operations, as well as arithmetic operations.
    """

    data: Tensor
    """
    The underlying data for the ``Buffer`` type.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.data, Tensor):
            raise TypeError(
                f"Expected data to be of type Tensor, got {type(self.data)=}"
            )

    def __iter__(self) -> Iterator[Tensor]:
        return iter(self.data)

    def __getitem__(self, idx):
        if not isinstance(idx, Tensor):
            idx = torch.tensor(idx, device=self.device)

        return type(self)(self.data[idx])

    __invert__ = _unary_op(operator.invert)
    __neg__ = _unary_op(operator.neg)

    __eq__ = _binary_op(operator.eq)
    __ne__ = _binary_op(operator.ne)
    __ge__ = _binary_op(operator.ge)
    __gt__ = _binary_op(operator.gt)
    __le__ = _binary_op(operator.le)
    __lt__ = _binary_op(operator.lt)

    __add__ = _binary_op(operator.add)
    __sub__ = _binary_op(operator.sub)
    __mul__ = _binary_op(operator.mul)
    __truediv__ = _binary_op(operator.truediv)
    __floordiv__ = _binary_op(operator.floordiv)
    __mod__ = _binary_op(operator.mod)
    __pow__ = _binary_op(operator.pow)

    max = _unary_op(lambda data: data.max().item())
    min = _unary_op(lambda data: data.min().item())
    all = _unary_op(lambda data: data.all().item())
    any = _unary_op(lambda data: data.any().item())
    count = _unary_op(lambda data: len(data))

    def map(self, op: Callable[[Tensor], Tensor], /) -> Self:
        return type(self)(op(self.data))

    def reduce[S](self, function: Callable[[Tensor, S], S], init: S) -> S:
        result = init

        for tensor in self:
            result = function(tensor, result)

        return result

    def select(self, idx: list[int] | Tensor) -> Self:
        return type(self)(self.data[idx])

    def to(self, device: str) -> Self:
        return type(self)(self.data.to(device))

    def size(self) -> tuple[int, ...]:
        return self.data.size()

    def to_pandas(self) -> Series:
        data = self.data.cpu()
        arr = np.array(data)
        return Series(arr)

    def datatype(self):
        return utils.batched_tensor_to_aioway_dtype(self.data)

    @property
    def device(self):
        return self.data.device
