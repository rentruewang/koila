# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import logging
import typing
from abc import ABC
from typing import Any, Literal, Self

import numpy as np
import torch
from numpy import dtype as _NumpyDType
from torch import dtype as _TorchDType

from aioway._errors import AiowayError

__all__ = [
    "DType",
    "dtype",
    "InfoDType",
    "StringDType",
    "TorchDType",
    "NumpyDtype",
]

LOGGER = logging.getLogger(__name__)

type _SupportedDtypes = Literal["info", "string", "torch", "numpy"]


class DType(ABC):
    r"""
    ``DType`` is a common base class supporting converting to and from
    an intermediate representation in ``aioway``, effectively supporting
    comparison and conversion between different frameworks.

    How it works::

        ``DType1().parse()`` -> gives something like ``InfoDType("type 1")``.
        ``DType2().parse()`` -> gives something like ``InfoDType("type 2")``.

        Then we can compare the ``InfoDType``s.

    Note:

        See :ref:`InfoDType` for the actual intermediate representation.

    """

    @typing.override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DType):
            return NotImplemented

        return self.parse() == other.parse()

    @abc.abstractmethod
    def parse(self) -> "InfoDType":
        """
        Extract the representation that can be used for comparing across frameworks.
        """

    @classmethod
    @abc.abstractmethod
    def create(cls, info: "InfoDType") -> Self:
        """
        Create the ``DType`` instance from the ``info`` object.
        """

    def torch(self) -> "TorchDType":
        return self._as_type(TorchDType)

    def numpy(self) -> "NumpyDtype":
        return self._as_type(NumpyDtype)

    def string(self) -> "StringDType":
        return self._as_type(StringDType)

    def info(self) -> "InfoDType":
        return self.parse()

    def _as_type(self, target_dtype: type["DType"], /) -> Any:
        return target_dtype.create(self.parse())


def dtype(dtype: str, kind: _SupportedDtypes = "string") -> DType:
    """
    This is the factory method for ``DType``,
    responsible for getting the subclasses based on ``kind``,
    and instantiate by parsing the ``dtype`` parameter.
    """
    common_repr = StringDType(dtype).parse()
    return _get_dtype_class(kind).create(common_repr)


def _get_dtype_class(kind: _SupportedDtypes) -> type[DType]:
    match kind:
        case "info":
            return InfoDType
        case "string":
            return StringDType
        case "torch":
            return TorchDType
        case "numpy":
            return NumpyDtype


@typing.final
@dcls.dataclass(frozen=True, eq=False)
class InfoDType(DType):
    r"""
    The universal data format used for comparing against other ``DType``s.

    Drawback::

        Since ``InfoDType`` is a product type rather than a sum type,
        it's easy to perform comparison, but does not support ``bool``,
        as ``bits`` would need special handling for ``bool``,
        because ``bool{k}`` doesn't make sense.
    """

    family: str
    """
    The family of numbers we are dealing with.
    """

    bits: int
    """
    How many bits are used in memory.
    """

    def __post_init__(self) -> None:
        if self.family not in ["int", "uint", "float", "complex"]:
            raise NotImplementedError

        if self.bits < 0:
            raise NotImplementedError

    def __str__(self):
        return f"{self.family}{self.bits}"

    @typing.override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, InfoDType):
            return self.family == other.family and self.bits == other.bits

        if isinstance(other, DType):
            return self == other.parse()

        return NotImplemented

    @typing.override
    def parse(self) -> "InfoDType":
        return self

    @classmethod
    @typing.override
    def create(cls, info: "InfoDType") -> "InfoDType":
        return info


@typing.final
@dcls.dataclass(frozen=True, eq=False)
class StringDType(DType):
    """
    The universal data format used for comparing against other ``DType``s.
    """

    dtype: str
    """
    The string representation of datatype.
    """

    @typing.override
    def parse(self) -> InfoDType:
        families = "int", "uint", "float", "complex"
        for family in families:
            if self.dtype.startswith(family):
                break
        else:
            # Only enter this if ``break`` is not triggered.
            raise UnsupportedDTypeError(f"'{self.dtype}' is invalid.")

        bits = int(self.dtype[len(family) :])
        return InfoDType(family=family, bits=bits)

    @classmethod
    @typing.override
    def create(cls, info: InfoDType) -> Self:
        return cls(str(info))


@typing.final
@dcls.dataclass(frozen=True, eq=False)
class TorchDType(DType):
    dtype: _TorchDType
    """
    The backing dtype from torch.
    """

    @typing.override
    def parse(self) -> InfoDType:
        return InfoDType(
            family=self._parse_family(), bits=itemsize_to_bits(self.dtype.itemsize)
        )

    def _parse_family(self):
        if self.dtype.is_complex:
            return "complex"
        elif self.dtype.is_floating_point:
            return "float"
        elif self.dtype.is_signed:
            return "int"
        else:
            return "uint"

    @classmethod
    @typing.override
    def create(cls, info: InfoDType) -> Self:
        "Create the ``TorchDType`` instance from ``DTypeInfo``."

        # ``torch.bool8`` is not a thing, but it is not supported now (rentruewang/aioway#110).
        # For ``family = "int"``, ``precison = 2``, this gives ``torch.int16``.
        return cls(getattr(torch, str(info)))


@typing.final
@dcls.dataclass(frozen=True, eq=False)
class NumpyDtype(DType):
    dtype: _NumpyDType
    """
    The backing dtype from torch.
    """

    @typing.override
    def parse(self) -> InfoDType:
        return InfoDType(family=self._parse_family(), bits=self.dtype.itemsize)

    def isdtype(self, kind: str) -> bool:
        """
        Short hand for ``np.isdtype(self.dtype, kind)``.
        """
        return np.isdtype(self.dtype, kind)

    def _parse_family(self):
        if self.isdtype("signed integer"):
            return "int"
        elif self.isdtype("unsigned integer"):
            return "uint"
        elif self.isdtype("real floating"):
            return "float"
        elif self.isdtype("complex floating"):
            return "complex"
        else:
            raise NotImplementedError

    @classmethod
    @typing.override
    def create(cls, info: InfoDType) -> Self:
        "Create the ``TorchDType`` instance from ``DTypeInfo``."

        # For ``family = "int"``, ``precison = 2``, this gives ``torch.int16``.
        return cls(np.dtype(f"{info.family}{itemsize_to_bits(info.bits)}"))


def itemsize_to_bits(itemsize: int) -> int:
    """
    Convert ``itemsize`` to number of bits, used in dtype suffix.

    For example::

        torch.int1.itemsize == 1
        torch.int8.itemsize == 1
        torch.int16.itemsize == 2
    """

    # For example, ``torch.int1`` has the same size as ``torch.int8``
    return 2 ** (itemsize - 2)


class UnsupportedDTypeError(AiowayError, ValueError, NotImplementedError): ...
