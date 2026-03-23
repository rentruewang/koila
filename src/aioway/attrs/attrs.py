# Copyright (c) AIoWay Authors - All Rights Reserved

"Schema is a collection of metadata describing the 'type' of data."

import dataclasses as dcls
import operator
import typing
from typing import Self

from torch import Tensor

from aioway import _logging
from aioway._typing import AnyUFunc2, UFunc1

from ._terms import Term
from .devices import Device, DeviceLike
from .dtypes import DType, DTypeLike
from .shapes import Shape, ShapeLike

__all__ = ["Attr", "attr", "AttrTerm", "AttrTermRhs"]


LOGGER = _logging.get_logger(__name__)

type AttrTermRhs = AttrTerm | Attr | Tensor | int | float | bool


@dcls.dataclass(frozen=True)
class Attr:
    """
    The "type' for a `Tensor`, describing everything we want to know about it.
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

    def __repr__(self):
        return repr(
            {
                "shape": self.shape,
                "dtype": self.dtype,
                "device": self.device,
            }
        )

    @property
    def term(self):
        return AttrTerm(self)

    @staticmethod
    def parse(device: DeviceLike, dtype: DTypeLike, shape: ShapeLike) -> Attr:
        "Alias for `attr` s.t. you don't need to import it."
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
    The convenient constructor for `Attr`.

    Args:
        device: Things that can be converted to `Device`.
        dtype: Things that can be converted to `DType`.
        shape: Things that can be converted to `Shape`.

    Returns:
        An attribute instance.
    """

    return Attr(
        device=Device.parse(device),
        dtype=DType.parse(dtype),
        shape=Shape.parse(shape),
    )


@dcls.dataclass(frozen=True)
class AttrTerm(Term[Attr]):
    attr: Attr

    def __post_init__(self):
        LOGGER.debug("Term for attr=%s created.", self.attr)

    def __repr__(self):
        return f"{self.attr!r}.term"

    @typing.no_type_check
    @LOGGER.function("DEBUG")
    def __invert__(self) -> Self:
        return self.__ufunc_op1(operator.invert)

    @typing.no_type_check
    @LOGGER.function("DEBUG")
    def __neg__(self) -> Self:
        return self.__ufunc_op1(operator.neg)

    @LOGGER.function("DEBUG")
    def __getitem__(self, key: int | Tensor | Attr | AttrTerm) -> Self:
        match key:
            case int():
                return self.make(
                    attr(device=self.device, dtype=self.dtype, shape=self.shape[1:])
                )
            case Attr() | AttrTerm() | Tensor():
                return self.make(
                    attr(
                        device=self.device,
                        dtype=self.dtype,
                        shape=Shape.parse(*key.shape, *self.shape[1:]),
                    )
                )

    @LOGGER.function("DEBUG")
    def __add__(self, other: AttrTermRhs) -> Self:
        return self.__ufunc_op2(other, operator.add)

    @LOGGER.function("DEBUG")
    def __sub__(self, other: AttrTermRhs) -> Self:
        return self.__ufunc_op2(other, operator.sub)

    @LOGGER.function("DEBUG")
    def __mul__(self, other: AttrTermRhs) -> Self:
        return self.__ufunc_op2(other, operator.mul)

    @LOGGER.function("DEBUG")
    def __truediv__(self, other: AttrTermRhs) -> Self:
        return self.__ufunc_op2(other, operator.truediv)

    @LOGGER.function("DEBUG")
    def __floordiv__(self, other: AttrTermRhs) -> Self:
        return self.__ufunc_op2(other, operator.floordiv)

    @LOGGER.function("DEBUG")
    def __mod__(self, other: AttrTermRhs) -> Self:
        return self.__ufunc_op2(other, operator.mod)

    @LOGGER.function("DEBUG")
    def __pow__(self, other: AttrTermRhs) -> Self:
        return self.__ufunc_op2(other, operator.pow)

    @typing.no_type_check
    @LOGGER.function("DEBUG")
    def __eq__(self, other: AttrTermRhs) -> Self:
        return self.__ufunc_op2(other, operator.eq)

    @typing.no_type_check
    @LOGGER.function("DEBUG")
    def __ne__(self, other: AttrTermRhs) -> Self:
        return self.__ufunc_op2(other, operator.ne)

    @typing.no_type_check
    @LOGGER.function("DEBUG")
    def __ge__(self, other: AttrTermRhs) -> Self:
        return self.__ufunc_op2(other, operator.ge)

    @typing.no_type_check
    @LOGGER.function("DEBUG")
    def __gt__(self, other: AttrTermRhs) -> Self:
        return self.__ufunc_op2(other, operator.gt)

    @typing.no_type_check
    @LOGGER.function("DEBUG")
    def __le__(self, other: AttrTermRhs) -> Self:
        return self.__ufunc_op2(other, operator.le)

    @typing.no_type_check
    @LOGGER.function("DEBUG")
    def __lt__(self, other: AttrTermRhs) -> Self:
        return self.__ufunc_op2(other, operator.lt)

    def __ufunc_op1(self, op: UFunc1) -> Self:
        return self.make(
            Attr(
                device=op(self.device.term).unpack(),
                shape=op(self.shape.term).unpack(),
                dtype=op(self.dtype.term).unpack(),
            )
        )

    def __ufunc_op2(self, other: AttrTermRhs, op: AnyUFunc2):
        match other:
            case AttrTerm():
                return self.make(
                    Attr(
                        device=op(self.device.term, other.device.term).unpack(),
                        shape=op(self.shape.term, other.shape.term).unpack(),
                        dtype=op(self.dtype.term, other.dtype.term).unpack(),
                    )
                )
            case Attr():
                return self.__ufunc_op2(other=other.term, op=op)
            case Tensor():
                return self.__ufunc_op2(other=Attr.from_tensor(other).term, op=op)
            case int() | float() | bool():
                t: type[int] | type[float] | type[bool] = type(other)
                return self.make(
                    Attr(
                        device=self.device,
                        shape=self.shape,
                        dtype=op(self.dtype.term, DType.parse(t)).unpack(),
                    )
                )

        raise TypeError(f"Do not know how to handle {type(other)=}.")

    @property
    def device(self):
        return self.attr.device

    @property
    def dtype(self):
        return self.attr.dtype

    @property
    def shape(self):
        return self.attr.shape

    def unpack(self) -> Attr:
        return self.attr

    @classmethod
    def make(cls, data: Attr, /) -> Self:
        return cls(data)
