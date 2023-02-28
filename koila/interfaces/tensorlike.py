from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, TypeVar

from .components import Arithmetic, Indexible, MemoryInfo, WithBatch

Number = TypeVar("Number", int, float)
Numeric = TypeVar("Numeric", int, float, bool)


class TensorLike(Arithmetic, Indexible, MemoryInfo, Protocol):
    """
    TensorLike is a protocol that mimics PyTorch's Tensor.

    TODO: TensorLike should also be mocking a PyTorch Tensor.
    """

    data: TensorLike
    "The underlying data that backs the tensor."

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
        """
        Retrieve the underlying 0-d data.

        Returns
        -------

        A boolean or an integer or a floating point number.
        """

        ...

    @abstractmethod
    def transpose(self, dim0: int, dim1: int) -> TensorLike:
        """
        Transposes swaps the order of the axises.

        Parameters
        ----------

        `dim0: int,` `dim1: int`
            The axises to swap.

        Returns
        -------

        A `TensorLike` that with the given axises swapped.
        """

        ...

    @property
    def T(self) -> TensorLike:
        """
        The matrix operator T. It is equivalent to transposing the first and second axises.

        Returns
        -------

        A `TensorLike` with first and second axises swapped.
        """

        return self.transpose(0, 1)

    @property
    @abstractmethod
    def requires_grad(self) -> bool:
        """
        If true, operations performed on this tensor will be added to the gradient tape.

        Returns
        -------

        A boolean indicating if gradient recording is on.
        """

        ...

    @abstractmethod
    def backward(self) -> None:
        """
        Performs a backward pass when called. Only available if the tensor is recording gradient.
        """

        ...


class BatchedTensorLike(TensorLike, WithBatch, Protocol):
    """
    `BatchedTensorLike` is `TensorLike` that adheres to the `WithBatch` protocol.
    """
