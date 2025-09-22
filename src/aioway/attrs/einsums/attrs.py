# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import functools
import logging
import re
import typing
from abc import ABC
from re import Pattern
from typing import Protocol

from aioway._errors import AiowayError
from aioway.attrs.devices import Device
from aioway.attrs.dtypes import DType
from aioway.attrs.shapes import Shape

from .parsers import EinsumSignature

__all__ = ["EinsumAttr", "EinsumName", "EinsumShape", "EinsumDType", "EinsumDevice"]

LOGGER = logging.getLogger(__name__)


@dcls.dataclass(eq=False, frozen=True, repr=False)
class EinsumAttr[T](ABC):
    """
    This class exists to share some common functionality between the ``Einsum*`` classes.
    """

    einsum: EinsumSignature
    """
    The einsum instance to use.
    """

    def __repr__(self) -> str:
        return repr(self.einsum)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return self.einsum == other.einsum

        if isinstance(other, str | EinsumSignature):
            return self.einsum == other

        return NotImplemented

    def __call__(self, items: tuple[T, ...], /) -> tuple[T, ...]:
        try:
            return self._call(items)
        except RuntimeError as re:
            raise EinsumCallError("Execution failed.") from re

    def _call(self, items: tuple[T, ...], /) -> tuple[T, ...]:

        if len(items) != self.einsum.num_inputs:
            raise RuntimeError

        result = self._compute(items)

        if len(result) != self.einsum.num_outputs:
            raise RuntimeError

        return result

    @abc.abstractmethod
    def _compute(self, items: tuple[T, ...], /) -> tuple[T, ...]:
        """
        Compute the target items, be it shape, dtype, device, etc.

        If failing, subclass should raise ``RuntimeError``.
        """
        ...

    @property
    def num_inputs(self):
        return self.einsum.num_inputs

    @property
    def num_outputs(self):
        return self.einsum.num_outputs

    @property
    def params(self):
        return self.einsum.params

    @property
    def results(self):
        return self.einsum.results

    @property
    def in_out(self):
        return self.einsum.in_out


@dcls.dataclass(eq=False, frozen=True, repr=False)
class EinsumName(EinsumAttr[str]):
    """
    ``EinsumSet`` represents the case where both inputs and outputs of ``Einsum`` are unique.
    """

    def __post_init__(self) -> None:
        LOGGER.debug("Checking an `EinsumMap`.")

        self.__check_in_or_out("parameters", self.params)
        self.__check_in_or_out("results", self.results)

    def __check_in_or_out(self, stage: str, items: tuple[str, ...]) -> None:
        # Check uniqueness
        if len(set(items)) != len(items):
            raise IllegalEinsumDeviceError(f"Duplicate {stage} in {items}")

        # Check null values.
        if not all(items):
            raise IllegalEinsumDeviceError(f"Null {stage} contained in {items}.")

    @typing.override
    def _compute(self, items: tuple[str, ...]) -> tuple[str, ...]:
        if items != self.params:
            raise RuntimeError

        return self.results


@dcls.dataclass(eq=False, frozen=True, repr=False)
class EinsumShape(EinsumAttr[Shape]):
    """
    Shape variant of ``Einsum``.
    """

    def __post_init__(self) -> None:
        LOGGER.debug("Checking an `EinsumShape`.")

        self.__check_all_dims_are_ascii()

    def __check_all_dims_are_ascii(self) -> None:
        regex = ascii_regex()

        for param in self.in_out():
            if regex.fullmatch(param):
                continue

            raise IllegalEinsumShapeError(f"Shape dimension: '{param}' is illegal.")

    @typing.override
    def _compute(self, items: tuple[Shape, ...]) -> tuple[Shape, ...]:
        if len(items) != len(self.params):
            raise RuntimeError

        mapping: dict[str, int] = {}

        def register_or_raise(key: str, val: int, /) -> None:
            if key in mapping and mapping[key] != val:
                raise RuntimeError

            mapping[key] = val

        for item, param in zip(items, self.params):
            if len(item) != len(param):
                raise RuntimeError

            for dim_key, dim_size in zip(param, item):
                register_or_raise(dim_key, dim_size)

        def convert_to_shape(param: str) -> Shape:
            return Shape.from_iterable(mapping[key] for key in param)

        return tuple(convert_to_shape(res) for res in self.results)


@functools.cache
def ascii_regex() -> Pattern[str]:
    return re.compile("[a-zA-Z]*")


@dcls.dataclass(eq=False, frozen=True, repr=False)
class EinsumDType(EinsumAttr[DType]):
    """
    DType variant of ``Einsum``.

    Currently, generic type are not allowed in ``EinsumDType`` s.t. the implemenatation is simple.
    This can change with the addition of operators like ``+-*/``.
    """

    def __post_init__(self):
        LOGGER.debug("Checking an `EinsumDType`.")

        _check_in_out_parse(self, DType, IllegalEinsumDTypeError, "dtype")

    @typing.override
    def _compute(self, items: tuple[DType, ...]) -> tuple[DType, ...]:
        if any(item != param for item, param in zip(items, self.params)):
            raise RuntimeError

        return tuple(DType.parse(res) for res in self.results)


@dcls.dataclass(eq=False, frozen=True, repr=False)
class EinsumDevice(EinsumAttr[Device]):
    """
    Device variant of ``Einsum``.

    Currently, generic type are not allowed in ``EinsumDevice`` s.t. the implemenatation is simple.
    This can change with the addition of operators like ``+-*/``.
    """

    def __post_init__(self):
        LOGGER.debug("Checking an `EinsumDevice`.")

        _check_in_out_parse(self, Device, IllegalEinsumDeviceError, "device")
        self._disallow_empty_in_or_out()

    def _disallow_empty_in_or_out(self) -> None:
        if not all(self.in_out()):
            raise IllegalEinsumDeviceError("Device cannot be omitted.")

    @typing.override
    def _compute(self, items: tuple[Device, ...]) -> tuple[Device, ...]:
        if any(item != param for item, param in zip(items, self.params)):
            raise RuntimeError

        return tuple(Device.parse(res) for res in self.results)


class _Parser(Protocol):
    def parse(self, item): ...


def _check_in_out_parse(
    einsum: EinsumAttr, parser: _Parser, error_type: type[ValueError], kind: str
) -> None:

    for val in einsum.in_out():
        try:
            # Because in `Einsum`, the null default is "" which is different from `Attr`s.
            parser.parse(val or None)
        except NotImplementedError as ne:
            raise error_type(f"The {kind} passed in: {val} is not parsable.") from ne


class EinsumCallError(AiowayError, RuntimeError): ...


class IllegalEinsumMappingError(AiowayError, ValueError): ...


class IllegalEinsumShapeError(AiowayError, ValueError): ...


class IllegalEinsumDTypeError(AiowayError, ValueError): ...


class IllegalEinsumDeviceError(AiowayError, ValueError): ...
