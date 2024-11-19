# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import logging
from collections.abc import Callable
from enum import Enum
from typing import ParamSpec

from .arrays import ArrayDtype
from .primitives import BoolDtype, FloatDtype, IntDtype
from .types import DataType

__all__ = ["DataTypeEnum"]

P = ParamSpec("P")
LOGGER = logging.getLogger(__name__)


class DataTypeEnum(Enum):
    """
    ``DataTypeFactory`` is a enum / builder for ``DataType``s.
    """

    BOOL = BoolDtype
    """
    The boolean data type.
    Boolean is represented by a bit, but takes up 1 byte because of memory layout.
    """

    INT = IntDtype
    """
    The integer data type.
    Integers can be 8, 16, 32, 64, or arbitrary length.
    """

    FLOAT = FloatDtype
    """
    The floating point data type.
    Floating point numbers can be 16, 32, or 64 bits.
    """

    ARRAY = ArrayDtype
    """
    The array data type.
    Arrays are able to cast to any other data type,
    and its dimensions are implicit (depending on the underlying data type).
    """

    def __getitem__(self, *args: P.args, **kwargs: P.kwargs) -> Callable[[], DataType]:
        LOGGER.debug(
            "Handle for %s with args = %s and kwargs = %s created.", self, args, kwargs
        )

        dtype = self.value
        return lambda: dtype(*args, **kwargs)

    def __call__(self) -> DataType:
        return self.value()
