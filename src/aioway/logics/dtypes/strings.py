# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
import typing
from typing import TypeVar

from .dtypes import DataType

_T = TypeVar("_T")


@typing.final
@dcls.dataclass(frozen=True)
class StrDtype(DataType):
    """
    The text data type.
    """

    length: int | None = None

    def accept(self, visitor: DataType.Visitor[_T]) -> _T:
        return visitor.string(self)
