from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, TypeVar

from .components import Arithmetic, Indexible, MemoryInfo, WithBatch

Number = TypeVar("Number", int, float)
Numeric = TypeVar("Numeric", int, float, bool)


class TensorLike(Arithmetic, Indexible, MemoryInfo, Protocol):
    """
    TensorLike is a protocol that mimics PyTorch's Tensor.
    """

    data: TensorLike

    @abstractmethod
    def __str__(self) -> str:
        ...

    def __bool__(self) -> bool:
        return bool(self.item())

    def __int__(self) -> int:
        return int(self.item())

    def __float__(self) -> float:
        return float(self.item())

    @abstractmethod
    def item(self) -> Numeric:
        ...

    @abstractmethod
    def transpose(self, dim0: int, dim1: int) -> TensorLike:
        ...

    @property
    def T(self) -> TensorLike:
        return self.transpose(0, 1)

    @abstractmethod
    def backward(self) -> None:
        ...


class BatchedTensorLike(TensorLike, WithBatch, Protocol):
    pass
