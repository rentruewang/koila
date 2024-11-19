# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
import typing
from typing import TypeVar

from .primitives import FloatDtype, PrimitiveDtype
from .types import DataType, DataTypeVisitor

__all__ = ["ArrayDtype"]

T = TypeVar("T")


@typing.final
@dcls.dataclass(eq=False, frozen=True, repr=False)
class ArrayDtype(DataType):
    """
    The array data type.
    """

    _shape: tuple[int, ...] = ()
    _dtype: PrimitiveDtype = dcls.field(default_factory=lambda: FloatDtype(32))

    def __repr__(self) -> str:
        shape = self._shape
        dtype = self._dtype
        return f"array({shape=}, {dtype=})"

    def __len__(self) -> int:
        return len(self._shape)

    def __getitem__(self, idx: int) -> int:
        return self._shape[idx]

    def _size(self) -> tuple[int, ...]:
        return self._shape

    def bytes(self) -> int:
        return self.numel() * self._dtype.bytes()

    def accept(self, visitor: DataTypeVisitor[T]) -> T:
        return visitor.array(self)

    @property
    def dtype(self) -> str:
        return repr(self._dtype)

    @property
    def ndim(self) -> int:
        return len(self._shape) if self._shape is not None else -1
