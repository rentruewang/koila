# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import inspect
import operator
from collections.abc import Callable, Iterator
from types import FunctionType
from typing import Self

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from pandas import Series
from torch import Tensor

from aioway.errors import AiowayError
from aioway.schemas import ArrayDtype

from ..blocks.clonable import Clonable
from . import utils

__all__ = ["Buffer"]


def _try_catch[T](func: Callable[[], T]) -> T:
    """
    Try and catching error as an ``aioway`` internal error.

    Raises:
        TensorComputeError:
            As ``torch`` raises ``RuntimeError`` upon erroring,
            if a torch runtime error is found, this is raised instead.

            This would leverage the ``inspect.getsource`` utility
            to get the original source of the function,
            as the original function are most likely ``lambda``s.

    Returns:
        The original output.
    """
    try:
        return func()
    except RuntimeError as re:
        source = inspect.getsource(func).strip()
        raise TensorComputeError(f"Function call: {source} failed") from re


def _unary_op[S: "Buffer"](op: Callable[[Tensor], Tensor]) -> FunctionType:
    def func(self: S) -> S:
        return _try_catch(lambda: type(self)(op(self.data)))

    assert isinstance(func, FunctionType)
    return func


type _RHS = int | float | bool | NDArray | Tensor


def _binary_op[S: "Buffer"](op: Callable[[Tensor, _RHS], Tensor]) -> FunctionType:
    def func(self: S, other: S | _RHS) -> S:
        if not isinstance(other, Buffer):
            other_data = other
        else:
            other_data = other.data

        return _try_catch(lambda: type(self)(data=op(self.data, other_data)))

    assert isinstance(func, FunctionType)
    return func


def _unary_tensor_op[S: "Buffer", T](op: Callable[[Tensor], T]) -> FunctionType:
    def func(self: S) -> T:
        return _try_catch(lambda: op(self.data))

    assert isinstance(func, FunctionType)
    return func


@dcls.dataclass(frozen=True)
class Buffer(Clonable):
    """
    ``Buffer`` is a thin wrapper for `torch.Tensor`,
    while providing additional checks,
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

    def __len__(self) -> int:
        return self.count()

    def __getitem__(self, idx: NDArray):
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

    count = _unary_tensor_op(lambda data: len(data))

    def map(self, op: Callable[[Tensor], Tensor], /) -> Self:
        return type(self)(op(self.data))

    def reduce[S](self, function: Callable[[Tensor, S], S], init: S) -> S:
        result = init

        for tensor in self:
            result = function(tensor, result)

        return result

    select = __getitem__

    def size(self) -> tuple[int, ...]:
        return self.data.size()

    @property
    def shape(self) -> tuple[int, ...]:
        return self.size()

    def datatype(self) -> ArrayDtype:
        return utils.tensor_dtype(self.data)

    def dtype(self) -> str:
        return str(self.datatype())

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def device(self):
        return self.data.device

    def _clone(self, *, device):
        return type(self)(data=self.data.clone().to(device=device))

    def to_pandas(self) -> Series:
        data = self.data.cpu()
        arr = np.array(data)
        return Series(arr)

    @classmethod
    def from_numpy(cls, arr: ArrayLike) -> Self:
        arr = np.array(arr)
        tensor = torch.from_numpy(arr)
        return cls(tensor)


class TensorComputeError(AiowayError, RuntimeError): ...
