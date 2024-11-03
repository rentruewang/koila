# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
import typing
from typing import Literal, TypeVar

from .dtypes import DataType, DataTypeVisitor

_T = TypeVar("_T")


@typing.final
@dcls.dataclass(frozen=True)
class BoolDtype(DataType):
    """
    The boolean data type.
    """

    def accept(self, visitor: DataTypeVisitor[_T]) -> _T:
        return visitor.boolean(self)


@typing.final
@dcls.dataclass(frozen=True)
class IntDtype(DataType):
    """
    The integer data type.
    """

    precision: Literal[8, 16, 32, 64] = 64

    def accept(self, visitor: DataTypeVisitor[_T]) -> _T:
        return visitor.integer(self)


@typing.final
@dcls.dataclass(frozen=True)
class FloatDtype(DataType):
    """
    The floating point data type.
    """

    precision: Literal[16, 32, 64] = 32

    def accept(self, visitor: DataTypeVisitor[_T]) -> _T:
        return visitor.floating(self)
