# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from collections.abc import Callable
from enum import Enum
from typing import ParamSpec, TypeVar

from .arrays import ArrayDtype
from .dtypes import DataType
from .primitives import BoolDtype, FloatDtype, IntDtype
from .strings import StrDtype

_T = TypeVar("_T")
_P = ParamSpec("_P")


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
        dtype = self.value
        return lambda: dtype(*args, **kwargs)

    def __call__(self) -> DataType:
        return self.value()


DtypeLike = DataType | Callable[[], DataType]
