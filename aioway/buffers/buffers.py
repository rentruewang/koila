# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import typing
from abc import ABC
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from torch import Tensor

from aioway.typings import Castable, Caster, Slicer

__all__ = ["Buffer"]


class Buffer(Castable, ABC):
    """
    ``Buffer`` is a thin wrapper for the underlying data structures,
    providing additional checks, and interfacing with ``aioway``.

    It handles indexing operations, as well as arithmetic operations.
    """

    @abc.abstractmethod
    def __len__(self) -> int: ...

    def __getitem__(self, idx: slice | ArrayLike) -> Self:
        if isinstance(idx, slice):
            slicer = Slicer(len(self))
            idx = slicer(idx)
            return self._getitem_slice(idx)

        idx = np.array(idx)
        return self._getitem_array(idx)

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
    def _getitem_slice(self, idx: slice) -> Self: ...

    @abc.abstractmethod
    def _getitem_array(self, idx: NDArray) -> Self: ...

    @abc.abstractmethod
    def _size(self) -> tuple[int, ...]: ...

    @property
    def shape(self) -> tuple[int, ...]:
        return self.size()

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @classmethod
    def _caster(cls) -> Caster:
        from .numpy import NumpyBuffer
        from .torch import TorchBuffer

        def tensor_to_array(buf: TorchBuffer) -> NumpyBuffer:
            return NumpyBuffer(buf.numpy())

        def array_to_tensor(buf: NumpyBuffer) -> TorchBuffer:
            return TorchBuffer(buf.torch())

        return Caster(
            base=Buffer,
            aliases=["torch", "numpy"],
            klasses=[TorchBuffer, NumpyBuffer],
            matrix=[
                [None, tensor_to_array],
                [array_to_tensor, None],
            ],
        )
