# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing
from collections.abc import Iterable, Iterator, Sequence
from typing import Self

import numpy as np
from numpy import ndarray as _NumpyNDArray
from numpy.typing import NDArray
from torch import Size

from aioway import _logging, _typing

from ._terms import Term

__all__ = ["ShapeLike", "Shape", "shape"]

LOGGER = _logging.get_logger(__name__)


type _PrimitiveNumber = float | int | bool
type _IntArrayLike = tuple[int, ...] | list[int] | NDArray[np.int_]
type ShapeCmpType = Shape | Size | _IntArrayLike | _PrimitiveNumber


_is_tuple_of_int = _typing.is_tuple_of(int)
_is_list_of_int = _typing.is_list_of(int)


@dcls.dataclass(frozen=True)
class Shape(Sequence[int]):
    """
    `Shape` represents a regular (non-jagged) array's dimensions,
    must be a `tuple` like object, and `tuple` would be used on it.

    Right now, it represents the shape of a `Tensor` **outside** the batch dimension.
    """

    dims: tuple[int, ...] = ()
    """
    The dimensions
    """

    def __post_init__(self):
        if not self.valid():
            raise ValueError(self)

        LOGGER.debug("Shape created: %s", self)

    @typing.override
    def __hash__(self) -> int:
        return hash(self.dims)

    @typing.override
    def __repr__(self) -> str:
        return str(self)

    @typing.override
    def __str__(self) -> str:
        dims_str = ", ".join(map(str, self.dims))
        return f"[{dims_str}]"

    @typing.override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Shape):
            return self.dims == other.dims

        if isinstance(other, Size):
            return other == self.dims

        # Do the numpy check first as `isinstance` is cheaper than the following ones.
        if isinstance(other, _NumpyNDArray):
            arr = np.array(self)
            return arr.ndim == other.ndim and np.all(arr == other).item()

        if _is_tuple_of_int(other) or _is_list_of_int(other):
            return tuple(self.dims) == tuple(other)

        return NotImplemented

    @typing.override
    def __len__(self):
        return len(self.dims)

    @typing.overload
    def __getitem__(self, idx: int) -> int: ...

    @typing.overload
    def __getitem__(self, idx: slice) -> Self: ...

    @typing.override
    def __getitem__(self, idx):
        match idx:
            case int():
                return self.dims[idx]
            case slice():
                return type(self)(self.dims[idx])
            case _:
                raise TypeError(type(idx))

    @typing.override
    def __iter__(self) -> Iterator[int]:
        return iter(self.dims)

    def __array__(self) -> NDArray:
        return np.array(self.dims)

    def valid(self) -> bool:
        """
        Check if the shape is a valid `Shape`.
        """

        LOGGER.debug("Checking if %s is a `Shape`", self)

        return _typing.is_seq_of(int)(self.dims)

    def valid_dims(self, dims: list[int]) -> bool:
        """
        Validate the dimensions amongst the shapes.
        """

        is_list_of_int = _typing.is_list_of(int)
        length = len(self)

        return is_list_of_int(dims) and (max(dims) >= length or min(dims) < -length)

    def wrap_dims(self, dims: list[int]) -> list[int]:
        # Make the dims positive.
        return [d % len(self) for d in dims]

    @property
    def ndim(self) -> int:
        """
        Number of dimensions in a shape.
        """
        return len(self)

    @property
    def size(self) -> int:
        """
        Number of elements in a shape.
        """

        total = 1

        for s in self:
            total *= s

        return total

    @property
    def term(self):
        return ShapeTerm.make(self)

    @typing.overload
    @staticmethod
    def parse(*dims: int) -> Shape: ...

    @typing.overload
    @staticmethod
    def parse(dim: ShapeLike, /) -> Shape: ...

    @staticmethod
    def parse(*dims):
        "Alias to the `shape` function so you won't need to import it."
        return shape(*dims)


type ShapeLike = int | Iterable[int] | Shape
"Types convertible to `Shape`s. Note that `int` can be converted as well."


@typing.overload
def shape(*dims: int) -> Shape: ...


@typing.overload
def shape(dim: ShapeLike, /) -> Shape: ...


def shape(*dims) -> Shape:
    """
    Convenience constructor for `Shape`.

    Takes either of the following signature:

    1. `shape(*dims)`. Here dims must be integers.
    2. `shape(iterable)`. Here dims must be iterable. No additional args.
    3. `shape(Shape)`. Returns it by reference.
    """

    try:
        # `shape(*int)`.
        if _is_tuple_of_int(dims):
            return _shape(dims)

        # `shape(iterable)`.
        elif len(dims) == 1:
            return _shape(dims[0])

        raise ValueError
    except ValueError:
        raise ValueError(*dims)


def _shape(dims) -> Shape:
    "Try converting dims to `Shape`, raise `ValueError` on failure."
    if isinstance(dims, Shape):
        return dims

    if isinstance(dims, Iterable):
        dims_tuple = tuple(dims)

        if _is_tuple_of_int(dims_tuple):
            return Shape(dims_tuple)

    raise ValueError


@dcls.dataclass(frozen=True)
class ShapeTerm(Term[Shape]):
    shape: Shape

    def __array__(self):
        return np.array(self.shape)

    def __invert__(self) -> Self:
        return self

    def __neg__(self) -> Self:
        return self

    def __add__(self, other: Self) -> Self:
        return self._broadcast_shapes(other)

    def __sub__(self, other: Self) -> Self:
        return self._broadcast_shapes(other)

    def __mul__(self, other: Self) -> Self:
        return self._broadcast_shapes(other)

    def __truediv__(self, other: Self) -> Self:
        return self._broadcast_shapes(other)

    def __floordiv__(self, other: Self) -> Self:
        return self._broadcast_shapes(other)

    def __mod__(self, other: Self) -> Self:
        return self._broadcast_shapes(other)

    def __pow__(self, other: Self) -> Self:
        return self._broadcast_shapes(other)

    @typing.no_type_check
    def __eq__(self, other: Self) -> Self:
        return self._broadcast_shapes(other)

    @typing.no_type_check
    def __ne__(self, other: Self) -> Self:
        return self._broadcast_shapes(other)

    def __ge__(self, other: Self) -> Self:
        return self._broadcast_shapes(other)

    def __gt__(self, other: Self) -> Self:
        return self._broadcast_shapes(other)

    def __le__(self, other: Self) -> Self:
        return self._broadcast_shapes(other)

    def __lt__(self, other: Self) -> Self:
        return self._broadcast_shapes(other)

    def _broadcast_shapes(self, other: Self):
        return type(self)(Shape(np.broadcast_shapes(self.shape, other.shape)))

    def unpack(self) -> Shape:
        return self.shape

    @classmethod
    def make(cls, data: Shape) -> Self:
        return cls(data)
