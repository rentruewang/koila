# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import abc
import typing
from typing import Protocol, TypeVar

_T = TypeVar("_T", covariant=True)

if typing.TYPE_CHECKING:
    from .arrays import ArrayDtype
    from .primitives import BoolDtype, FloatDtype, IntDtype
    from .types import DataType


class DataTypeVisitor(Protocol[_T]):
    def visit(self, dtype: "DataType", /) -> _T:
        return dtype.accept(self)

    @abc.abstractmethod
    def boolean(self, dtype: "BoolDtype", /) -> _T: ...

    @abc.abstractmethod
    def integer(self, dtype: "IntDtype", /) -> _T: ...

    @abc.abstractmethod
    def floating(self, dtype: "FloatDtype", /) -> _T: ...

    @abc.abstractmethod
    def array(self, dtype: "ArrayDtype", /) -> _T: ...
