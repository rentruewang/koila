from __future__ import annotations

import functools
import operator
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Tuple, TypeVar, Union, overload, runtime_checkable

from torch import Tensor
from torch import device as Device
from torch import dtype as DType

from . import constants

T = TypeVar("T", covariant=True)
V = TypeVar("V", contravariant=True)


@runtime_checkable
class Runnable(Protocol[T]):
    @abstractmethod
    def run(self, partial: PartialInfo | None = None) -> T:
        ...


class QueryType(Enum):
    Numel = "numel"
    Memory = "memory"


@runtime_checkable
class TensorMixin(Protocol):
    @overload
    @abstractmethod
    def size(self) -> Tuple[int, ...]:
        ...

    @overload
    @abstractmethod
    def size(self, dim: int) -> int:
        ...

    @abstractmethod
    def size(self, dim: int | None = None) -> int | Tuple[int, ...]:
        ...

    def numel(self) -> int:
        return functools.reduce(operator.mul, self.size(), 1)

    def dim(self) -> int:
        return len(self.size())

    @abstractmethod
    def dtype(self) -> DType:
        return self.metadata().dtype

    @abstractmethod
    def device(self) -> str | Device:
        return self.metadata().device

    @abstractmethod
    def metadata(self) -> MetaData:
        return MetaData(self.dtype(), self.device())


@runtime_checkable
class RunnableTensor(Runnable[Tensor], TensorMixin, Protocol):
    @abstractmethod
    def batch(self) -> int | None:
        ...


@dataclass(frozen=True)
class MetaData:
    dtype: DType
    device: str | Device
    batch: int | None = None


def dtype(tensor: TensorLike) -> DType:
    if isinstance(tensor, Tensor):
        return tensor.dtype

    return tensor.dtype()


def device(tensor: TensorLike) -> str | Device:
    if isinstance(tensor, Tensor):
        return tensor.device

    return tensor.device()


def metadata(tensor: TensorLike) -> MetaData:
    if isinstance(tensor, Tensor):
        return MetaData(tensor.dtype, tensor.device)

    return tensor.metadata()


def memory(tensor: TensorLike) -> int:
    if isinstance(tensor, Tensor):
        return tensor.numel() * constants.MEMORY_BYTES[tensor.dtype]

    return tensor.numel() * constants.MEMORY_BYTES[tensor.dtype()]


TensorLike = Union[Tensor, RunnableTensor]
