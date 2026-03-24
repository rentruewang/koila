# Copyright (c) AIoWay Authors - All Rights Reserved

"The implementation for dtypes, supports different backends."

import dataclasses as dcls
import functools
import re
import typing
from typing import Any, Literal, Self

import numpy as np
import torch
from numpy import dtype as NumpyDType
from torch import dtype as TorchDType

from aioway._ops import OpSign
from aioway._tracking import ModuleApiTracker, logging

from ._terms import Term

__all__ = ["DType", "dtype", "DTypeLike"]

LOGGER = logging.get_logger(__name__)
TRACKER = ModuleApiTracker(lambda: DType)

type DTypeFamily = Literal["int", "float", "bool"]
"""
The DType strings family type.
"""


class DType:
    r"""
    `DType` is a class supporting converting to and from
    its string representation in `aioway`, effectively supporting
    comparison and conversion between different frameworks.
    """

    def __init__(self, family: DTypeFamily, bits: int):
        if bits % 8 != 0:
            raise ValueError(f"Bits should be multiple of 8. Got bits={self._bits}.")

        self._family: DTypeFamily = family
        self._bits: int = bits

    def __str__(self):
        """
        Get the representation of the type, must be the most specialized.

        For example, the output should not be "float" due to ambiguity,
        but rather "float32" or "float64" etc.
        """

        match f := self._family:
            case "bool":
                return "bool"
            case "int" | "float":
                return f"{f}{self._bits}"
            case _:
                raise NotImplementedError(self._family)

    def __hash__(self) -> int:
        return hash(str(self))

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: Any) -> bool:
        try:
            parsed = dtype(other)
        except ValueError:
            return NotImplemented

        # Parsing sucessful.
        return self.family == parsed.family and self.bits == parsed.bits

    @property
    def family(self) -> DTypeFamily:
        """
        The family of the dtype. For example, "float64"'s family is "float".
        """

        return self._family

    @property
    def bits(self) -> int:
        """
        The width of the dtype in bytes. Greater or equal to 1.
        """
        return self._bits

    def numpy(self) -> NumpyDType:
        "Convert this to a numpy dtype."
        return np.dtype(str(self))

    def torch(self) -> TorchDType:
        "Convert this to a torch dtype."
        return getattr(torch, str(self))

    @property
    def term(self):
        return DTypeTerm.make(self)

    @classmethod
    def boolean(cls) -> Self:
        return cls(family="bool", bits=8)

    @staticmethod
    def parse(item: DTypeLike):
        "Alias to the `dtype` function so you don't need to import it."
        return dtype(item)


type _PrimitiveType = type[int] | type[float] | type[bool]
type DTypeLike = str | DType | _PrimitiveType | TorchDType | NumpyDType
"Types that can be converted to `Dtype` with the public `dtype` function (or `DType.parse`)."


def dtype(dtype: DTypeLike, /) -> DType:
    """
    The convenient wrapper to create a `DType` from compatible types.

    Raises:
        ValueError: If we don't know how to handle the dtype.
    """

    if isinstance(dtype, DType):
        return dtype

    # Handling the basic types.
    if isinstance(dtype, type):
        return _parse_primitive_type(dtype)

    # Convert with regex.
    if isinstance(dtype, str):
        return _parse_regex(dtype)

    if isinstance(dtype, TorchDType):
        return _parse_torch(dtype)

    if isinstance(dtype, NumpyDType):
        return _parse_numpy(dtype)

    raise ValueError(f"Not sure how to handle {dtype=}.")


def _parse_primitive_type(dtype: type):
    if dtype == int:
        return DType("int", 64)

    if dtype == float:
        return DType("float", 32)

    if dtype == bool:
        return DType("bool", 8)

    raise ValueError(dtype)


def _parse_regex(dtype: str, /) -> DType:
    """
    Create the `DType` instance from the `info` object.

    Raises:
        ValueError: If the dtyep cannot be parsed.
    """

    if m := _int_dtype().match(dtype):
        _, bits = m.groups()
        return DType(family="int", bits=int(bits) if bits else 64)

    if m := _float_dtype().match(dtype):
        _, bits = m.groups()
        return DType(family="float", bits=int(bits) if bits else 32)

    if m := _bool_dtype().match(dtype):
        return DType(family="bool", bits=8)

    raise ValueError(dtype)


def _parse_torch(dtype: TorchDType, /) -> DType:
    "Create a `Dtype` from a `torch.dtype`."
    if dtype == torch.bool:
        return DType.boolean()

    if dtype.is_complex:
        raise ValueError("We do not handle complex types yet!")

    # Itemsize correspond to bytes.
    bits = dtype.itemsize * 8

    # Both complex and bool are handled above.
    family: DTypeFamily = "float" if dtype.is_floating_point else "int"

    return DType(family=family, bits=bits)


def _parse_numpy(dtype: NumpyDType, /) -> DType:
    family = _parse_numpy_family(dtype)
    bits = _parse_numpy_bits(dtype, family)

    return DType(family=family, bits=bits)


def _parse_numpy_family(dtype: NumpyDType, /) -> DTypeFamily:
    "Create a `Dtype` from a `numpy.dtype`."
    if np.isdtype(dtype, "integral"):
        return "int"

    if np.isdtype(dtype, "real floating"):
        return "float"

    if np.isdtype(dtype, "bool"):
        return "bool"

    raise ValueError(f"Cannot handle numpy {dtype=}.")


def _parse_numpy_bits(dtype: NumpyDType, family: DTypeFamily, /):
    match family:
        case "bool":
            return 8
        case "int":
            return np.iinfo(dtype).bits
        case "float":
            return np.finfo(dtype).bits

    raise ValueError(f"Unhandled {family=}")


@functools.cache
def _float_dtype():
    return re.compile(r"^(float)(16|32|64|128)?$", re.IGNORECASE)


@functools.cache
def _int_dtype():
    return re.compile(r"^(int)(8|16|32|64|128)?$", re.IGNORECASE)


@functools.cache
def _bool_dtype():
    return re.compile(r"^(bool)$", re.IGNORECASE)


@dcls.dataclass(frozen=True)
class DTypeTerm(Term[DType]):
    dtype: DType

    def __invert__(self):
        return self.__identical("__invert__")

    def __neg__(self):
        return self.__identical("__neg__")

    def __add__(self, other: Self | DTypeLike) -> Self:
        return self.__broadcast(self.dtype, other, name="__add__")

    def __sub__(self, other: Self | DTypeLike) -> Self:
        return self.__broadcast(self.dtype, other, name="__sub__")

    def __mul__(self, other: Self | DTypeLike) -> Self:
        return self.__broadcast(self.dtype, other, name="__mul__")

    def __truediv__(self, other: Self | DTypeLike) -> Self:
        return self.__broadcast(self.dtype, other, name="__truediv__")

    def __floordiv__(self, other: Self | DTypeLike) -> Self:
        return self.__broadcast(self.dtype, other, name="__floordiv__")

    def __mod__(self, other: Self | DTypeLike) -> Self:
        return self.__broadcast(self.dtype, other, name="__mod__")

    def __pow__(self, other: Self | DTypeLike) -> Self:
        return self.__broadcast(self.dtype, other, name="__pow__")

    @typing.no_type_check
    def __eq__(self, other: Self | DTypeLike):
        return self.__boolean(other, name="eq")

    @typing.no_type_check
    def __ne__(self, other: Self | DTypeLike):
        return self.__boolean(other, name="ne")

    def __ge__(self, other: Self | DTypeLike):
        return self.__boolean(other, name="ge")

    def __gt__(self, other: Self | DTypeLike):
        return self.__boolean(other, name="gt")

    def __le__(self, other: Self | DTypeLike):
        return self.__boolean(other, name="le")

    def __lt__(self, other: Self | DTypeLike):
        return self.__boolean(other, name="lt")

    def unpack(self) -> DType:
        return self.dtype

    def __identical(self, name: str) -> Self:
        with TRACKER(name=name, signature=OpSign(DType, DType)):
            return self

    @classmethod
    def __broadcast(cls, left: DType, right: Self | DTypeLike, name: str) -> Self:
        with TRACKER(name=name, signature=OpSign(DType, DType, DType)):
            return cls.__broadcast_impl(left, right)

    @classmethod
    def __broadcast_impl(cls, left: DType, right: Self | DTypeLike) -> Self:
        try:
            rhs = DType.parse(_as_dtype(right))
        except ValueError:
            return NotImplemented

        np_lhs = left.numpy()
        np_rhs = rhs.numpy()

        promoted = np.result_type(np_lhs, np_rhs)
        return cls.make(DType.parse(promoted))

    @classmethod
    def __boolean(cls, right: Self | DTypeLike, name: str):
        with TRACKER(name=name, signature=OpSign(DType, DType, DType)):
            return cls.__boolean_impl(right=right)

    @classmethod
    def __boolean_impl(cls, right: Self | DTypeLike):
        try:
            _ = DType.parse(_as_dtype(right))
        except ValueError:
            return NotImplemented

        return cls.make(DType.boolean())

    @classmethod
    def make(cls, data: DType) -> Self:
        return cls(data)


def _as_dtype(item: DTypeTerm | DTypeLike, /) -> DTypeLike:
    return item.dtype if isinstance(item, DTypeTerm) else item
