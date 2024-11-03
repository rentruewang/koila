# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from abc import ABC
from typing import Protocol, TypeVar

if typing.TYPE_CHECKING:
    from .arrays import ArrayDtype
    from .primitives import BoolDtype, FloatDtype, IntDtype
    from .strings import StrDtype

_T = TypeVar("_T", covariant=True)


@dcls.dataclass(frozen=True)
class DataType(ABC):
    """
    ``DataType`` is the base class of all ``aioway`` data types.
    It is an empty base class, designed to be extended by concrete classes
    and used with `Dtype.Visitor`, acting like tagged unions.
    """

    class Visitor(Protocol[_T]):
        def __call__(self, dtype: "DataType", /) -> _T:
            return dtype.accept(self)

        @abc.abstractmethod
        def boolean(self, dtype: "BoolDtype", /) -> _T: ...

        @abc.abstractmethod
        def integer(self, dtype: "IntDtype", /) -> _T: ...

        @abc.abstractmethod
        def floating(self, dtype: "FloatDtype", /) -> _T: ...

        @abc.abstractmethod
        def array(self, dtype: "ArrayDtype", /) -> _T: ...

        @abc.abstractmethod
        def string(self, dtype: "StrDtype", /) -> _T: ...

    @abc.abstractmethod
    def accept(self, visitor: Visitor[_T], /) -> _T: ...
