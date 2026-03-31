# Copyright (c) AIoWay Authors - All Rights Reserved

"Schema is a collection of metadata describing the 'type' of data."

import dataclasses as dcls
import operator
import typing

import torch

from aioway import fake
from aioway._signs import Signature
from aioway._tracking import ModuleApiTracker, logging
from aioway._typing import AnyUFunc2, IntArray, UFunc1

from .devices import Device, DeviceLike
from .dtypes import DType, DTypeLike
from .shapes import Shape, ShapeLike

__all__ = ["Attr", "AttrTerm", "AttrTermRhs"]


LOGGER = logging.get_logger(__name__)
TRACKER = ModuleApiTracker(lambda: Attr)

type AttrTermRhs = AttrTerm | Attr | torch.Tensor | int | float | bool


@dcls.dataclass(frozen=True)
class Attr:
    """
    The "type' for a `torch.Tensor`, describing everything we want to know about it.
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

    def to_tensor(self):
        """
        Generate a random tensor.
        This should be used under fake mode.
        """

        return (
            torch.empty(self.shape.concrete())
            .to(self.device.torch())
            .to(self.dtype.torch())
        )

    @classmethod
    def parse(
        cls, device: DeviceLike, dtype: DTypeLike, shape: ShapeLike
    ) -> typing.Self:
        """
        The convenient constructor for `Attr`.

        Args:
            device: Things that can be converted to `Device`.
            dtype: Things that can be converted to `DType`.
            shape: Things that can be converted to `Shape`.

        Returns:
            An attribute instance.
        """

        return cls(
            device=Device.parse(device),
            dtype=DType.parse(dtype),
            shape=Shape.parse(shape),
        )

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, /) -> typing.Self:
        "Parse the `torch.Tensor`'s `Attr` representation"

        return cls.parse(
            device=tensor.device,
            shape=tensor.shape,
            dtype=tensor.dtype,
        )


@dcls.dataclass(frozen=True)
class AttrTerm:
    attr: Attr

    def __post_init__(self):
        LOGGER.debug("Term for attr=%s created.", self.attr)

    def __repr__(self):
        return f"{self.attr!r}.term"

    @typing.no_type_check
    def __invert__(self) -> typing.Self:
        return self.__ufunc_op1(operator.invert)

    @typing.no_type_check
    def __neg__(self) -> typing.Self:
        return self.__ufunc_op1(operator.neg)

    def __getitem__(
        self, key: int | slice | IntArray | torch.Tensor | Attr | AttrTerm
    ) -> typing.Self:
        sign = Signature(Attr, type(key), Attr)
        with TRACKER(name="__getitem__", signature=sign):
            return self.__getitem_impl(key)

    def __add__(self, other: AttrTermRhs) -> typing.Self:
        return self.__ufunc_op2(other, operator.add)

    def __sub__(self, other: AttrTermRhs) -> typing.Self:
        return self.__ufunc_op2(other, operator.sub)

    def __mul__(self, other: AttrTermRhs) -> typing.Self:
        return self.__ufunc_op2(other, operator.mul)

    def __truediv__(self, other: AttrTermRhs) -> typing.Self:
        return self.__ufunc_op2(other, operator.truediv)

    def __floordiv__(self, other: AttrTermRhs) -> typing.Self:
        return self.__ufunc_op2(other, operator.floordiv)

    def __mod__(self, other: AttrTermRhs) -> typing.Self:
        return self.__ufunc_op2(other, operator.mod)

    def __pow__(self, other: AttrTermRhs) -> typing.Self:
        return self.__ufunc_op2(other, operator.pow)

    @typing.no_type_check
    def __eq__(self, other: AttrTermRhs) -> typing.Self:
        return self.__ufunc_op2(other, operator.eq)

    @typing.no_type_check
    def __ne__(self, other: AttrTermRhs) -> typing.Self:
        return self.__ufunc_op2(other, operator.ne)

    @typing.no_type_check
    def __ge__(self, other: AttrTermRhs) -> typing.Self:
        return self.__ufunc_op2(other, operator.ge)

    @typing.no_type_check
    def __gt__(self, other: AttrTermRhs) -> typing.Self:
        return self.__ufunc_op2(other, operator.gt)

    @typing.no_type_check
    def __le__(self, other: AttrTermRhs) -> typing.Self:
        return self.__ufunc_op2(other, operator.le)

    @typing.no_type_check
    def __lt__(self, other: AttrTermRhs) -> typing.Self:
        return self.__ufunc_op2(other, operator.lt)

    @fake.enable_func
    def __ufunc_op1(self, op: UFunc1) -> typing.Self:
        signature = Signature(Attr, Attr)
        with TRACKER(name=f"__{op.__qualname__}__", signature=signature):
            return self.make(Attr.from_tensor(op(self.attr.to_tensor())))

    @fake.enable_func
    def __ufunc_op2(self, other: AttrTermRhs, op: AnyUFunc2):
        match other:
            case torch.Tensor():
                return self.__ufunc_op2_tensor(other=other, op=op)
            case AttrTerm():
                return self.__ufunc_op2(other=other.attr, op=op)
            case Attr():
                return self.__ufunc_op2(other=other.to_tensor(), op=op)
            case int() | float() | bool():
                return self.__ufunc_op2(other=torch.tensor(other), op=op)

        raise TypeError(f"Do not know how to handle {type(other)=}.")

    @fake.enable_func
    def __ufunc_op2_tensor(self, other: torch.Tensor, op: AnyUFunc2) -> typing.Self:
        signature = Signature(Attr, Attr, Attr)
        with TRACKER(name=f"__{op.__qualname__}__", signature=signature):
            return self.make(Attr.from_tensor(op(self.attr.to_tensor(), other)))

    @fake.enable_func
    def __getitem_impl(
        self, key: int | slice | IntArray | torch.Tensor | Attr | AttrTerm, /
    ):
        if isinstance(key, Attr):
            return self.__getitem_impl(key.to_tensor())

        if isinstance(key, AttrTerm):
            return self.__getitem_impl(key.attr)

        # Fake tensors will not work with real ones.
        if fake.is_real_tensor(key):
            key = fake.to_fake_tensor(key)

        return self.make(Attr.from_tensor(self.attr.to_tensor()[key]))

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
    def make(cls, data: Attr, /) -> typing.Self:
        return cls(data)
