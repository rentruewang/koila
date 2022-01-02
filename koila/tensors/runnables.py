from __future__ import annotations

from abc import abstractmethod
from typing import Callable, NamedTuple, Protocol, TypeVar, runtime_checkable

from .tensors import TensorLike

T = TypeVar("T", covariant=True)


@runtime_checkable
class Runnable(Protocol[T]):
    @abstractmethod
    def run(self) -> T:
        ...


class BatchedPair(NamedTuple):
    batch: int
    no_batch: int


class BatchInfo(NamedTuple):
    index: int
    value: int

    def map(self, func: Callable[[int], int]) -> BatchInfo:
        index = func(self.index)
        return BatchInfo(index, self.value)


class RunnableTensor(Runnable[TensorLike], Protocol):
    @abstractmethod
    def run(self, partial: range | None = None) -> TensorLike:
        ...
