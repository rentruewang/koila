# Copyright (c) AIoWay Authors - All Rights Reserved

"Schema is a collection of metadata describing the 'type' of data."

import dataclasses as dcls
import operator
import typing
from collections import abc as cabc

import torch

from aioway import _signs, _tracking, _typing, fake
from aioway._tracking import logging

from . import devices, dtypes, shapes

__all__ = ["Attr", "AttrTerm", "AttrTermRhs"]


LOGGER = logging.get_logger(__name__)
TRACKER = _tracking.get_tracker(lambda: Attr)

type AttrTermRhs = AttrTerm | Attr | torch.Tensor | int | float | bool


@dcls.dataclass(frozen=True)
class Attr:
    """
    The "type' for a `torch.Tensor`, describing everything we want to know about it.
    """

    device: devices.Device
    """
    The device for the column.
    """

    dtype: dtypes.DType
    """
    The data type for the column.
    """

    shape: shapes.Shape
    """
    The shape of individual items in the column.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.device, devices.Device):
            raise TypeError(type(self.device))

        if not isinstance(self.dtype, dtypes.DType):
            raise TypeError(type(self.dtype))

        if not isinstance(self.shape, shapes.Shape):
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
        cls,
        device: devices.DeviceLike,
        dtype: dtypes.DTypeLike,
        shape: shapes.ShapeLike,
    ) -> typing.Self:
        """
        The convenient constructor for `Attr`.

        Args:
            device: Things that can be converted to `devices.Device`.
            dtype: Things that can be converted to `dtypes.DType`.
            shape: Things that can be converted to `shapes.Shape`.

        Returns:
            An attribute instance.
        """

        return cls(
            device=devices.Device.parse(device),
            dtype=dtypes.DType.parse(dtype),
            shape=shapes.Shape.parse(shape),
        )

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, /) -> typing.Self:
        "Parse the `torch.Tensor`'s `Attr` representation"

        return cls.parse(
            device=tensor.device,
            shape=tensor.shape,
            dtype=tensor.dtype,
        )


class AttrDict(typing.TypedDict):
    device: devices.DeviceLike
    dtype: dtypes.DTypeLike
    shape: shapes.ShapeLike


@typing.runtime_checkable
class AttrProto(typing.Protocol):
    device: devices.DeviceLike
    dtype: dtypes.DTypeLike
    shape: shapes.ShapeLike


type AttrLike = Attr | AttrDict


def attr(item: AttrLike, /) -> Attr:
    "The convenient constructor function for `Attr` to convert from similar types."

    if isinstance(item, Attr):
        return item

    if _is_attr_dict(item):
        return Attr.parse(
            device=item["device"],
            shape=item["shape"],
            dtype=item["dtype"],
        )

    if isinstance(item, AttrProto):
        return Attr.parse(
            device=item.device,
            dtype=item.dtype,
            shape=item.shape,
        )

    raise TypeError(f"Do not know how to handle {item=}, because it is malformed.")


@typing.no_type_check
def _is_attr_dict(item: object) -> typing.TypeGuard[AttrDict]:

    if not isinstance(item, cabc.Mapping):
        return False

    try:
        _ = Attr.parse(
            device=item["device"],
            dtype=item["dtype"],
            shape=item["shape"],
        )
    except Exception:
        return False

    return True


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
        self, key: int | slice | _typing.IntArray | torch.Tensor | Attr | AttrTerm
    ) -> typing.Self:
        sign = _signs.Signature(Attr, type(key), Attr)
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
    def __ufunc_op1(self, op: _typing.UFunc1) -> typing.Self:
        signature = _signs.Signature(Attr, Attr)
        with TRACKER(name=f"__{op.__qualname__}__", signature=signature):
            return self.make(Attr.from_tensor(op(self.attr.to_tensor())))

    @fake.enable_func
    def __ufunc_op2(self, other: AttrTermRhs, op: _typing.AnyUFunc2):
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
    def __ufunc_op2_tensor(
        self, other: torch.Tensor, op: _typing.AnyUFunc2
    ) -> typing.Self:
        signature = _signs.Signature(Attr, Attr, Attr)
        with TRACKER(name=f"__{op.__qualname__}__", signature=signature):
            return self.make(Attr.from_tensor(op(self.attr.to_tensor(), other)))

    @fake.enable_func
    def __getitem_impl(
        self, key: int | slice | _typing.IntArray | torch.Tensor | Attr | AttrTerm, /
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
