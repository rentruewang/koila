# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import typing
from typing import Literal

from .types import DataType, DataTypeVisitor

__all__ = ["BoolDtype", "FloatDtype", "IntDtype"]


@dcls.dataclass(eq=False, frozen=True, repr=False)
class PrimitiveDtype(DataType):
    def _size(self) -> tuple[()]:
        return ()


@typing.final
@dcls.dataclass(eq=False, frozen=True, repr=False)
class BoolDtype(PrimitiveDtype):
    """
    The boolean data type.
    """

    def __repr__(self) -> str:
        return "bool"

    def bytes(self) -> int:
        return 1

    def accept[T](self, visitor: DataTypeVisitor[T]) -> T:
        return visitor.boolean(self)


@typing.final
@dcls.dataclass(eq=False, frozen=True, repr=False)
class IntDtype(PrimitiveDtype):
    """
    The integer data type.
    """

    precision: Literal[8, 16, 32, 64] = 64

    def __repr__(self) -> str:
        return f"int{self.precision}"

    def bytes(self) -> int:
        return self.precision // 8

    def accept[T](self, visitor: DataTypeVisitor[T]) -> T:
        return visitor.integer(self)


@typing.final
@dcls.dataclass(eq=False, frozen=True, repr=False)
class FloatDtype(PrimitiveDtype):
    """
    The floating point data type.
    """

    precision: Literal[16, 32, 64] = 32

    def __repr__(self) -> str:
        return f"float{self.precision}"

    def bytes(self) -> int:
        return self.precision // 8

    def accept[T](self, visitor: DataTypeVisitor[T]) -> T:
        return visitor.floating(self)
