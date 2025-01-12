# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import typing
from abc import ABC

from numpy.typing import NDArray
from torch import Tensor

__all__ = ["Buffer"]


class Buffer(ABC):
    @abc.abstractmethod
    def __len__(self) -> int: ...

    def __array__(self) -> NDArray:
        return self.numpy()

    @abc.abstractmethod
    def numpy(self) -> NDArray: ...

    @abc.abstractmethod
    def torch(self) -> Tensor: ...

    @typing.overload
    def size(self) -> tuple[int, ...]: ...

    @typing.overload
    def size(self, dim: int) -> int: ...

    def size(self, dim=-1):
        shape = self._size()
        return shape if dim < 0 else shape[dim]

    @abc.abstractmethod
    def _size(self, dim: int = -1) -> tuple[int, ...]: ...

    @property
    def shape(self) -> tuple[int, ...]:
        return self.size()

    @property
    def ndim(self) -> int:
        return len(self.shape)
