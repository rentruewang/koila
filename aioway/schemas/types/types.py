# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
import functools
import operator
import typing
from abc import ABC
from typing import Protocol

if typing.TYPE_CHECKING:
    from .arrays import ArrayDtype
    from .primitives import BoolDtype, FloatDtype, IntDtype

__all__ = ["DataType", "DataTypeVisitor"]


@dcls.dataclass(eq=False, frozen=True)
class DataType(ABC):
    """
    ``DataType`` is the base class of all ``aioway`` data types.
    It is an empty base class, designed to be extended by concrete classes
    and used with `DataTypeVisitor`, acting like tagged unions.

    Todo:
        Improve usability in both relational algebra domain and tensor domain.
    """

    def __hash__(self) -> int:
        return hash(repr(self))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return repr(self) == other

        if isinstance(other, type(self)):
            return self.__dict__ == other.__dict__

        return NotImplemented

    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @abc.abstractmethod
    def accept[T](self, visitor: "DataTypeVisitor[T]", /) -> T: ...

    @abc.abstractmethod
    def _size(self) -> tuple[int, ...]: ...

    @abc.abstractmethod
    def bytes(self) -> int: ...

    @typing.overload
    def size(self, dim: int) -> int: ...

    @typing.overload
    def size(self, dim: None = None) -> tuple[int, ...]: ...

    def size(self, dim: int | None = None) -> int | tuple[int, ...]:
        shape = self.shape

        if dim is None:
            return shape

        return shape[dim]

    def to_array(self) -> "ArrayDtype":
        from .arrays import ArrayDtype

        class ToArray(DataTypeVisitor[ArrayDtype]):
            def _to_array(self, dtype):
                return ArrayDtype((), dtype)

            integer = floating = boolean = _to_array

            def array(self, dtype: ArrayDtype) -> ArrayDtype:
                return dtype

        return ToArray().visit(self)

    def numel(self) -> int:
        return functools.reduce(operator.mul, self.shape, 1)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._size()


class DataTypeVisitor[T](Protocol):
    def visit(self, dtype: "DataType", /) -> T:
        return dtype.accept(self)

    @abc.abstractmethod
    def boolean(self, dtype: "BoolDtype", /) -> T: ...

    @abc.abstractmethod
    def integer(self, dtype: "IntDtype", /) -> T: ...

    @abc.abstractmethod
    def floating(self, dtype: "FloatDtype", /) -> T: ...

    @abc.abstractmethod
    def array(self, dtype: "ArrayDtype", /) -> T: ...
