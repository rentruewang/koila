from __future__ import annotations

import functools
import operator
from abc import abstractmethod
from typing import Protocol, Tuple, TypeVar, overload, runtime_checkable

from torch import Tensor

T = TypeVar("T", covariant=True)
V = TypeVar("V", contravariant=True)


@runtime_checkable
class Runnable(Protocol[T]):
    @abstractmethod
    def run(self) -> T:
        ...


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
