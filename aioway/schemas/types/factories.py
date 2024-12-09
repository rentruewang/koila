# Copyright (c) RenChu Wang - All Rights Reserved

import logging
from enum import Enum
from typing import Any

from .arrays import ArrayDtype
from .primitives import BoolDtype, FloatDtype, IntDtype
from .types import DataType

__all__ = ["DataTypeEnum"]

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

    def __call__(self, *args: Any, **kwargs: Any) -> DataType:
        LOGGER.debug(
            "Handle for %s with args = %s and kwargs = %s created.", self, args, kwargs
        )

        dtype = self.value
        return dtype(*args, **kwargs)
