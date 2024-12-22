# Copyright (c) RenChu Wang - All Rights Reserved

from __future__ import annotations

import functools
import operator
from abc import abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    NamedTuple,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    overload,
    runtime_checkable,
)

from torch import Tensor
from torch import device as Device
from torch import dtype as DType

from . import constants

E = TypeVar("E")
T = TypeVar("T", covariant=True)
V = TypeVar("V", contravariant=True)


@runtime_checkable
class Runnable(Protocol[T]):
    @abstractmethod
    def run(self) -> T: ...


@runtime_checkable
class TensorMixin(Protocol):
    @overload
    @abstractmethod
    def size(self) -> Tuple[int, ...]: ...

    @overload
    @abstractmethod
    def size(self, dim: int) -> int: ...

    @abstractmethod
    def size(self, dim: int | None = None) -> int | Tuple[int, ...]: ...

    def numel(self) -> int:
        return functools.reduce(operator.mul, self.size(), 1)

    def dim(self) -> int:
        return len(self.size())

    @abstractmethod
    def dtype(self) -> DType: ...

    @abstractmethod
    def device(self) -> str | Device: ...


class BatchedPair(NamedTuple):
    batch: int
    no_batch: int


class BatchInfo(NamedTuple):
    index: int
    value: int

    def map(self, func: Callable[[int], int]) -> BatchInfo:
        index = func(self.index)
        return BatchInfo(index, self.value)


@runtime_checkable
class RunnableTensor(Runnable[Tensor], TensorMixin, Protocol):
    @abstractmethod
    def batch(self) -> BatchInfo | None: ...

    @abstractmethod
    def run(self, partial: Tuple[int, int] | None = None) -> Tensor: ...

    @abstractmethod
    def visit(self, nodes: Dict[int, TensorLike]) -> None: ...

    def buffer(self) -> Dict[int, TensorLike]:
        nodes = {}
        self.visit(nodes)
        return nodes

    def buffer_numel(self) -> BatchedPair:
        buffer = self.buffer().values()
        return BatchedPair(
            sum(t.numel() for t in buffer if bat(t) is not None),
            sum(t.numel() for t in buffer if bat(t) is None),
        )

    def buffer_memory(self) -> BatchedPair:
        buffer = self.buffer().values()
        return BatchedPair(
            sum(mem(t) for t in buffer if bat(t) is not None),
            sum(mem(t) for t in buffer if bat(t) is None),
        )

    def memory(self) -> int:
        return mem(self)


def dtyp(tensor: TensorLike) -> DType:
    if isinstance(tensor, Tensor):
        return tensor.dtype

    return tensor.dtype()


def dev(tensor: TensorLike) -> str | Device:
    if isinstance(tensor, Tensor):
        return tensor.device

    return tensor.device()


def mem(tensor: TensorLike) -> int:
    dt = dtyp(tensor)
    numel = tensor.numel()

    if (batch := bat(tensor)) is not None:
        numel //= batch.value

    return constants.MEMORY_BYTES[dt] * numel


def bat(tensor: TensorLike) -> BatchInfo | None:
    if isinstance(tensor, RunnableTensor):
        return tensor.batch()
    return None


TensorLike = Union[Tensor, RunnableTensor]


@overload
def run(val: RunnableTensor, partial: Tuple[int, int] | None = None) -> Tensor: ...


@overload
def run(val: Runnable[E], partial: Tuple[int, int] | None = None) -> E: ...


@overload
def run(val: E, partial: Tuple[int, int] | None = None) -> E: ...


def run(val: Any, partial: Tuple[int, int] | None = None) -> Any:
    if isinstance(val, RunnableTensor):
        return val.run(partial)

    if isinstance(val, Runnable):
        return val.run()

    return val
