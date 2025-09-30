# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing
from typing import Literal

from aioway._errors import AiowayError

from .dtypes import DType

__all__ = ["IntDType", "FloatDType", "BoolDType"]


@dcls.dataclass(eq=False, frozen=True)
class _PrecisionMixin:
    precision: int
    """
    The precision of the dtype in terms of bits,
    equal to ``numpy``'s ``itemsize`` * 8.
    """

    def __post_init__(self) -> None:
        if self.precision <= 0:
            raise InvalidPrecisionError(f"Precision must be > 0. Got {self.precision}.")

        if self.precision % 8 != 0:
            raise InvalidPrecisionError(
                f"Precision must be a multiple of 8. Got {self.precision}."
            )


@dcls.dataclass(eq=False, frozen=True)
class IntDType(_PrecisionMixin, DType):
    """
    The integer dtype in ``aioway``.
    """

    precision: int
    """
    The precision of the integer dtype in terms of bits,
    equal to ``numpy``'s ``itemsize`` * 8.
    """

    @typing.override
    def __str__(self) -> str:
        return f"int{self.precision}"


@dcls.dataclass(eq=False, frozen=True)
class FloatDType(_PrecisionMixin, DType):
    """
    The floating point dtype in ``aioway``.
    """

    @typing.override
    def __str__(self) -> str:
        return f"float{self.precision}"


@dcls.dataclass(eq=False, frozen=True)
class BoolDType(DType):
    @typing.override
    def __str__(self) -> Literal["bool"]:
        return "bool"


class InvalidPrecisionError(AiowayError, ValueError): ...
