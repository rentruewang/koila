# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import logging
from collections.abc import Callable
from enum import Enum
from typing import ParamSpec

from .arrays import ArrayDtype
from .dtypes import DataType
from .primitives import BoolDtype, FloatDtype, IntDtype
from .strings import StrDtype

_P = ParamSpec("_P")
_LOGGER = logging.getLogger(__name__)


class DtypeFactory(Enum):
    """
    ``DataType`` is a enum / builder for ``Dtype``s.

    Todo:
        Remove the non-core types and make them extensions.
    """

    BOOL = BoolDtype
    """
    The boolean data type.
    Boolean is represented by a bit.
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

    STR = StrDtype
    """
    The text data type.
    Usually, they are encoded into embeddings during use.
    """

    ARRAY = ArrayDtype
    """
    The array data type.
    Arrays are able to cast to any other data type,
    and its dimensions are implicit (depending on the underlying data type).
    """

    def __getitem__(
        self, *args: _P.args, **kwargs: _P.kwargs
    ) -> Callable[[], DataType]:
        _LOGGER.debug(
            "Handle for %s with args = %s and kwargs = %s created.", self, args, kwargs
        )

        dtype = self.value
        return lambda: dtype(*args, **kwargs)

    def __call__(self) -> DataType:
        return self.value()
