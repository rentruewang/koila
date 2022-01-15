from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, TypeVar, runtime_checkable

from .tensorlike import BatchedTensorLike, TensorLike

T = TypeVar("T", covariant=True)


@runtime_checkable
class Runnable(Protocol[T]):
    @abstractmethod
    def run(self) -> T:
        ...


@runtime_checkable
class RunnableTensor(Runnable[TensorLike], BatchedTensorLike, Protocol):
    @abstractmethod
    def run(self, partial: range | None = None) -> TensorLike:
        ...
