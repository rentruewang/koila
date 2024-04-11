from __future__ import annotations

import abc
import typing
from typing import Protocol, TypeVar

from torch import Tensor

from koila.interfaces import DataType, TensorLike

T = TypeVar("T")


@typing.runtime_checkable
class Runnable(Protocol[T]):
    """
    A `Runnable` is something that evaluates to an expression after `run()` is called.
    """

    @abc.abstractmethod
    def run(self) -> T:
        """
        Run function evaluates the value that a `Runnable` represents.

        Returns:
            The value that the `Runnable` represents. Must be overridden.
        """

        ...


@typing.runtime_checkable
class RunnableTensor(Runnable[Tensor], TensorLike, DataType, Protocol):
    """
    A `RunnableTensor` is a `Tensor` that's `Runnable`.
    """

    @abc.abstractmethod
    def run(self, partial: range | None = None) -> Tensor:
        """
        A solid `Tensor` is expected after running a `RunnableTensor`.
        """

        ...
