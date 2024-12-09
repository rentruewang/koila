# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import typing

from .primitives import FloatDtype, PrimitiveDtype
from .types import DataType, DataTypeVisitor

__all__ = ["ArrayDtype"]


@typing.final
@dcls.dataclass(eq=False, frozen=True, repr=False)
class ArrayDtype(DataType):
    """
    The array data type.
    """

    shape: tuple[int, ...] = ()
    dtype: PrimitiveDtype = dcls.field(default_factory=lambda: FloatDtype(32))

    def __repr__(self) -> str:
        shape = self.shape
        dtype = self.dtype
        return f"array({shape=}, {dtype=})"

    def __len__(self) -> int:
        return len(self.shape)

    def __getitem__(self, idx: int) -> int:
        return self.shape[idx]

    def _size(self) -> tuple[int, ...]:
        return self.shape

    def bytes(self) -> int:
        return self.numel() * self.dtype.bytes()

    def accept[T](self, visitor: DataTypeVisitor[T]) -> T:
        return visitor.array(self)

    @property
    def ndim(self) -> int:
        return len(self.shape) if self.shape is not None else -1
