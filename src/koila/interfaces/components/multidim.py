from __future__ import annotations

import abc
import functools
import operator
import typing
from typing import Protocol


class MultiDimensional(Protocol):
    """
    A `MultiDimensional` tensor is an array that's multi-dimensional, as opposed to 1-dimensional like `list`s.
    """

    def __len__(self) -> int:
        return self.size(0)

    def dim(self) -> int:
        return len(self.size())

    @property
    def ndim(self) -> int:
        return self.dim()

    @property
    def ndimension(self) -> int:
        return self.ndim

    @typing.overload
    @abc.abstractmethod
    def size(self) -> tuple[int, ...]:
        ...

    @typing.overload
    @abc.abstractmethod
    def size(self, dim: int) -> int:
        ...

    @abc.abstractmethod
    def size(self, dim: int | None = None) -> int | tuple[int, ...]:
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        return self.size()

    def numel(self) -> int:
        return functools.reduce(operator.mul, self.shape, 1)
