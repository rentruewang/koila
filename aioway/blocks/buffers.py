# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
from abc import ABC
from collections.abc import Callable
from typing import Self

from pandas import Series
from torch import Tensor

from aioway.blocks._typing import TensorNumber
from aioway.schemas import DataType

__all__ = ["Buffer"]


@dcls.dataclass(frozen=True)
class Buffer(ABC):
    """
    ``Buffer`` represenets an in-memory column, which is a set of homogenius items,
    acting as the most primitive operation in ``aioway``.
    """

    def __len__(self) -> int:
        return self.count()

    @abc.abstractmethod
    def __getitem__(self, idx): ...

    @abc.abstractmethod
    def __invert__(self) -> Self: ...

    @abc.abstractmethod
    def __add__(self, other: Self | TensorNumber) -> Self: ...

    @abc.abstractmethod
    def __sub__(self, other: Self | TensorNumber) -> Self: ...

    @abc.abstractmethod
    def __mul__(self, other: Self | TensorNumber) -> Self: ...

    @abc.abstractmethod
    def __truediv__(self, other: Self | TensorNumber) -> Self: ...

    @abc.abstractmethod
    def __floordiv__(self, other: Self | TensorNumber) -> Self: ...

    @abc.abstractmethod
    def __mod__(self, other: Self | TensorNumber) -> Self: ...

    @abc.abstractmethod
    def __pow__(self, other: Self | TensorNumber) -> Self: ...

    @abc.abstractmethod
    def count(self) -> int: ...

    @abc.abstractmethod
    def max(self) -> int | float: ...

    @abc.abstractmethod
    def min(self) -> int | float: ...

    @abc.abstractmethod
    def reduce[S](self, function: Callable[[Tensor, S], S], init: S) -> S: ...

    @abc.abstractmethod
    def all(self) -> bool: ...

    @abc.abstractmethod
    def any(self) -> bool: ...

    @abc.abstractmethod
    def map(self, op: Callable[[Tensor], Tensor], /) -> Self: ...

    @abc.abstractmethod
    def select(self, idx: list[int]) -> Self: ...

    @abc.abstractmethod
    def size(self) -> tuple[int, ...]: ...

    @abc.abstractmethod
    def datatype(self) -> DataType: ...

    @abc.abstractmethod
    def to_pandas(self) -> Series: ...

    @property
    def shape(self) -> tuple[int, ...]:
        return self.size()

    @property
    def dtype(self) -> DataType:
        return self.datatype()
