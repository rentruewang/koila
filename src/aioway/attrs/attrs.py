# Copyright (c) AIoWay Authors - All Rights Reserved

"Schema is a collection of metadata describing the 'type' of data."

import dataclasses as dcls
import logging
import operator
import typing
from typing import Protocol, Self

from torch import Tensor

from ._terms import Term
from .devices import Device, DeviceLike, DeviceOperand
from .dtypes import DType, DTypeLike, DTypeTerm
from .shapes import Shape, ShapeLike, ShapeTerm

__all__ = ["Attr", "attr"]


LOGGER = logging.getLogger(__name__)


@dcls.dataclass(frozen=True)
class Attr:
    """
    Attributes for a single column in a ``Table``.
    """

    device: Device
    """
    The device for the column.
    """

    dtype: DType
    """
    The data type for the column.
    """

    shape: Shape
    """
    The shape of individual items in the column.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.device, Device):
            raise TypeError(type(self.device))

        if not isinstance(self.dtype, DType):
            raise TypeError(type(self.dtype))

        if not isinstance(self.shape, Shape):
            raise TypeError(type(self.shape))

    @property
    def term(self):
        return AttrTerm(self)

    @staticmethod
    def parse(device: DeviceLike, dtype: DTypeLike, shape: ShapeLike) -> Attr:
        "Alias for ``attr`` s.t. you don't need to import it."
        return attr(device=device, dtype=dtype, shape=shape)

    @staticmethod
    def from_tensor(tensor: Tensor, /) -> Attr:
        return attr(
            device=tensor.device,
            shape=tensor.shape,
            dtype=tensor.dtype,
        )


def attr(device: DeviceLike, dtype: DTypeLike, shape: ShapeLike) -> Attr:
    """
    The convenient constructor for ``Attr``.

    Args:
        device: Things that can be converted to ``Device``.
        dtype: Things that can be converted to ``DType``.
        shape: Things that can be converted to ``Shape``.

    Returns:
        An attribute instance.
    """

    return Attr(
        device=Device.parse(device),
        dtype=DType.parse(dtype),
        shape=Shape.parse(shape),
    )


class UFunc1(Protocol):
    def __call__[T](self, item: T, /) -> T: ...


class UFunc2(Protocol):
    def __call__[T](self, left: T, right: T, /) -> T: ...


@dcls.dataclass(frozen=True)
class AttrTerm(Term[Attr]):
    attr: Attr

    @typing.no_type_check
    def __invert__(self) -> Self:
        return self._apply_op1(operator.invert)

    @typing.no_type_check
    def __neg__(self) -> Self:
        return self._apply_op1(operator.neg)

    def __add__(self, other: Self) -> Self:
        return self._apply_op2(other, operator.add)

    def __sub__(self, other: Self) -> Self:
        return self._apply_op2(other, operator.sub)

    def __mul__(self, other: Self) -> Self:
        return self._apply_op2(other, operator.mul)

    def __truediv__(self, other: Self) -> Self:
        return self._apply_op2(other, operator.truediv)

    def __floordiv__(self, other: Self) -> Self:
        return self._apply_op2(other, operator.floordiv)

    def __mod__(self, other: Self) -> Self:
        return self._apply_op2(other, operator.mod)

    def __pow__(self, other: Self) -> Self:
        return self._apply_op2(other, operator.pow)

    @typing.no_type_check
    def __eq__(self, other: Self) -> Self:
        return self._apply_op2(other, operator.eq)

    @typing.no_type_check
    def __ne__(self, other: Self) -> Self:
        return self._apply_op2(other, operator.ne)

    @typing.no_type_check
    def __ge__(self, other: Self) -> Self:
        return self._apply_op2(other, operator.ge)

    @typing.no_type_check
    def __gt__(self, other: Self) -> Self:
        return self._apply_op2(other, operator.gt)

    @typing.no_type_check
    def __le__(self, other: Self) -> Self:
        return self._apply_op2(other, operator.le)

    @typing.no_type_check
    def __lt__(self, other: Self) -> Self:
        return self._apply_op2(other, operator.lt)

    def _apply_op1(self, op: UFunc1) -> Self:
        return self.make(
            Attr(
                device=op(self.device_op).unpack(),
                shape=op(self.shape_op).unpack(),
                dtype=op(self.dtype_op).unpack(),
            )
        )

    def _apply_op2(self, other: Self, op: UFunc2):
        return self.make(
            Attr(
                device=op(self.device_op, other.device_op).unpack(),
                shape=op(self.shape_op, other.shape_op).unpack(),
                dtype=op(self.dtype_op, other.dtype_op).unpack(),
            )
        )

    @property
    def device_op(self):
        return DeviceOperand.make(self.attr.device)

    @property
    def shape_op(self):
        return ShapeTerm.make(self.attr.shape)

    @property
    def dtype_op(self):
        return DTypeTerm.make(self.attr.dtype)

    def unpack(self) -> Attr:
        return self.attr

    @classmethod
    def make(cls, data: Attr) -> Self:
        return cls(data)
