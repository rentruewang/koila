from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, TypeVar, runtime_checkable

from torch import Tensor

from .tensorlike import BatchedTensorLike, TensorLike

T = TypeVar("T", covariant=True)


@runtime_checkable
class Runnable(Protocol[T]):
    """
    A `Runnable` is something that evaluates to an expression after `run()` is called.
    """

    @abstractmethod
    def run(self) -> T:
        """
        Run function evaluates the value that a `Runnable` represents.

        Returns
        -------

        The value that the `Runnable` represents. Must be overridden.
        """

        ...


@runtime_checkable
class RunnableTensor(Runnable[Tensor], BatchedTensorLike, Protocol):
    """
    A `RunnableTensor` is a `Tensor` that's `Runnable`.
    """

    @abstractmethod
    def run(self, partial: range | None = None) -> Tensor:
        """
        A solid `Tensor` is expected after running a `RunnableTensor`. 
        """

        ...
        
