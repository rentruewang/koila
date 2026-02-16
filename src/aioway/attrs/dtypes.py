# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import functools
import logging
import re
import typing
from abc import ABC
from typing import ClassVar, Literal, Self

import numpy as np
import torch
from numpy import dtype as _NumpyDType
from torch import dtype as _TorchDType

__all__ = [
    "DType",
    "dtype_with_kind",
    "dtype",
    "ComposedDType",
    "TorchDType",
    "NumpyDType",
]

LOGGER = logging.getLogger(__name__)


type DTypeBackend = Literal["composed", "torch", "numpy"]
"""
The backend of a dtype. Must be one of the following:

- composed: The dtype is backed by its family and bits.
- torch: The dtype has `torch.dtype` underlying dtype.
- numpy: The dtype has `np.dtype` underlying dtype.
"""

type DTypeFamily = Literal["int", "float", "bool"]
"""
The DType strings family type.
"""


class DType(ABC):
    r"""
    ``DType`` is a common base class supporting converting to and from
    its string representation in ``aioway``, effectively supporting
    comparison and conversion between different frameworks.
    """

    BACKEND: ClassVar[DTypeBackend]
    """
    The backend of a dtype.
    """

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Get the representation of the type, must be the most specialized.

        For example, the output should not be "float" due to ambiguity,
        but rather "float32" or "float64" etc.
        """

        ...

    @typing.override
    def __hash__(self) -> int:
        return hash(str(self))

    @typing.final
    @typing.override
    def __repr__(self) -> str:
        return f"{self.BACKEND}.{str(self)}"

    @typing.override
    def __eq__(self, other: object) -> bool:
        match other:
            case str():
                return str(self) == other

            # Invoke ``str`` for other if ``DType``
            case DType():
                return str(self) == str(other)

            # Convert ``self`` into a ``torch.dtype``,
            # using ``str`` as a medium to convert into ``torch``.
            case _TorchDType():
                return TorchDType.parse(str(self)).dtype == other

            # Don't know how to handle others.
            case _:
                return NotImplemented

    @property
    @abc.abstractmethod
    def family(self) -> DTypeFamily:
        """
        The family of the dtype. For example, "float64"'s family is "float".
        """

    @property
    @abc.abstractmethod
    def bits(self) -> int:
        """
        The width of the dtype in bytes. Greater or equal to 1.
        """

        ...

    @classmethod
    @abc.abstractmethod
    def parse(cls, dtype: str, /) -> Self:
        """
        Create the ``DType`` instance from the ``info`` object.

        Raises:
            ValueError: If the dtyep cannot be parsed.
        """


type DTypeLike = str | DType
"Types convertible to ``DType``."


def dtype(dtype: DTypeLike, /):
    match dtype:
        case DType():
            return dtype
        case str():
            return dtype_with_kind(dtype, kind="composed")

    raise TypeError(type(dtype))


def dtype_with_kind(dtype: str, /, kind: DTypeBackend) -> DType:
    """
    This is the factory method for ``DType``,
    responsible for getting the subclasses based on ``kind``,
    and instantiate by parsing the ``dtype`` parameter.
    """

    LOGGER.debug("Creating datatype: %s with '%s' backend", dtype, kind)
    result = _get_dtype_class(kind).parse(dtype)
    LOGGER.debug("Create datatype: %s", result)
    return result


def _get_dtype_class(kind: DTypeBackend) -> type[DType]:
    match kind:
        case "composed":
            return ComposedDType
        case "torch":
            return TorchDType
        case "numpy":
            return NumpyDType


@typing.final
class ComposedDType(DType):
    """
    The universal data format used for comparing against other ``DType``s.
    """

    BACKEND = "composed"

    def __init__(self, family: DTypeFamily, bits: int):
        if bits % 8 != 0:
            raise ValueError(f"Bits should be multiple of 8. Got {self._bits=}.")

        self._family: DTypeFamily = family
        self._bits: int = bits

    @typing.override
    def __str__(self):
        match f := self._family:
            case "bool":
                return "bool"
            case "int" | "float":
                return f"{f}{self._bits}"
            case _:
                raise NotImplementedError(self._family)

    @property
    @typing.override
    def family(self) -> DTypeFamily:
        return self._family

    @property
    @typing.override
    def bits(self) -> int:
        return self._bits

    @classmethod
    @typing.override
    def parse(cls, dtype: str, /) -> Self:
        if m := _int_dtype().match(dtype):
            _, bits = m.groups()
            return cls(family="int", bits=int(bits) if bits else 64)

        if m := _float_dtype().match(dtype):
            _, bits = m.groups()
            return cls(family="float", bits=int(bits) if bits else 32)

        if m := _bool_dtype().match(dtype):
            return cls(family="bool", bits=8)

        raise ValueError(dtype)


@functools.cache
def _float_dtype():
    return re.compile(r"^(float)(16|32|64|128)?$", re.IGNORECASE)


@functools.cache
def _int_dtype():
    return re.compile(r"^(int)(16|32|64|128)?$", re.IGNORECASE)


@functools.cache
def _bool_dtype():
    return re.compile(r"^(bool)$", re.IGNORECASE)


@typing.final
class TorchDType(DType):
    BACKEND = "torch"

    def __init__(self, dtype: _TorchDType) -> None:
        self._dtype = dtype

        try:
            self._check_torch_dtype()
        except (TypeError, ValueError) as e:
            raise ValueError(dtype) from e

    def __str__(self):
        return str(self._dtype).removeprefix("torch.")

    def _check_torch_dtype(self):
        if not isinstance(self._dtype, _TorchDType):
            raise TypeError(f"Must accept a {_TorchDType}. Got {type(self._dtype)}.")

        # Try getting the family.
        if not str(self._dtype).startswith(f"torch.{self._family}"):
            raise ValueError(self._dtype)

    @functools.cached_property
    def _family(self) -> DTypeFamily:
        if self._dtype == torch.bool:
            return "bool"

        if self._dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            return "int"

        if self._dtype in [torch.float16, torch.float32, torch.float64]:
            return "float"

        raise ValueError(self._dtype)

    @property
    @typing.override
    def family(self) -> DTypeFamily:
        return self._family

    @property
    @typing.override
    def bits(self) -> int:
        return self._dtype.itemsize * 8

    @property
    def dtype(self):
        return self._dtype

    @classmethod
    @typing.override
    def parse(cls, dtype: str, /) -> Self:
        "Create the ``TorchDType`` instance from ``DTypeInfo``."
        try:
            return cls(getattr(torch, dtype))
        except AttributeError as ae:
            raise ValueError(dtype) from ae


@typing.final
class NumpyDType(DType):
    """
    A ``DType`` backed by ``np.dtype``.
    """

    BACKEND = "numpy"

    def __init__(self, dtype: _NumpyDType) -> None:
        self._dtype = dtype

        try:
            self._check_numpy_dtype()
        except Exception as e:
            raise ValueError(dtype) from e

    @typing.override
    def __str__(self) -> str:
        return str(self._dtype)

    def _check_numpy_dtype(self):
        if not isinstance(self._dtype, _NumpyDType):
            raise TypeError(f"DType: {self._dtype} is not numpy.")

        if self._bits % 8 != 0:
            raise AssertionError("Bit should be a multiple of 8.")

    @property
    @typing.override
    def family(self) -> DTypeFamily:
        if np.isdtype(self._dtype, "integral"):
            return "int"

        if np.isdtype(self._dtype, "real floating"):
            return "float"

        if np.isdtype(self._dtype, "bool"):
            return "bool"

        raise ValueError(f"Cannot handle numpy dtype {self._dtype}")

    @property
    @typing.override
    def bits(self) -> int:
        return self._bits

    @functools.cached_property
    def _bits(self) -> int:
        match self.family:
            case "bool":
                return 8
            case "int":
                return np.iinfo(self._dtype).bits
            case "float":
                return np.finfo(self._dtype).bits

    @classmethod
    @typing.override
    def parse(cls, dtype: str, /) -> Self:
        "Create the ``TorchDType`` instance from ``DTypeInfo``."

        try:
            dt = np.dtype(dtype)
        except Exception as e:
            raise ValueError(dtype) from e

        return cls(dt)
