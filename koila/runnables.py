from __future__ import annotations

import functools
import operator
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Set, Tuple, TypeVar, Union, overload, runtime_checkable

from torch import Tensor
from torch import device as Device
from torch import dtype as DType

from . import constants

T = TypeVar("T", covariant=True)
V = TypeVar("V", contravariant=True)


@runtime_checkable
class Runnable(Protocol[T]):
    @abstractmethod
    def run(self) -> T:
        ...


class QueryType(Enum):
    Numel = "numel"
    Memory = "memory"


@runtime_checkable
class RunnableTensor(Runnable[Tensor], Protocol):
    @overload
    def size(self) -> Tuple[int, ...]:
        ...

    @overload
    def size(self, dim: int) -> int:
        ...

    def size(self, dim: int | None = None) -> int | Tuple[int, ...]:
        return self._size_impl(dim)

    @abstractmethod
    def _size_impl(self, dim: int | None = None) -> int | Tuple[int, ...]:
        ...

    def numel(self) -> int:
        return functools.reduce(operator.mul, self.size(), 1)

    def dim(self) -> int:
        return len(self.size())

    def upstream(self, query: QueryType = QueryType.Numel) -> int:
        visited: Set[TensorLike] = set()
        self.upstream_tensors(visited)

        if query == QueryType.Numel:
            return sum(tensor.numel() for tensor in visited)

        if QueryType.Memory:
            return sum(memory(tensor) for tensor in visited)

        raise ValueError

    @abstractmethod
    def upstream_tensors(self, visited: Set[TensorLike]) -> None:
        ...

    @abstractmethod
    def dtype(self) -> DType:
        return self.metadata().dtype

    @abstractmethod
    def device(self) -> str | Device:
        return self.metadata().device

    @abstractmethod
    def metadata(self) -> MetaData:
        return MetaData(self.dtype(), self.device())


@dataclass
class MetaData:
    dtype: DType
    device: str | Device


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
