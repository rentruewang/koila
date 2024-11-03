# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
import typing
from typing import TypeVar

from .dtypes import DataType
from .primitives import BoolDtype, FloatDtype, IntDtype

_T = TypeVar("_T")


@typing.final
@dcls.dataclass(frozen=True)
class ArrayDtype:
    """
    The array data type.
    """

    shape: tuple[int | None, ...] | None = None
    dtype: BoolDtype | IntDtype | FloatDtype | None = None

    def accept(self, visitor: DataType.Visitor[_T]) -> _T:
        return visitor.array(self)

    @property
    def ndim(self) -> int:
        return len(self.shape) if self.shape is not None else -1
