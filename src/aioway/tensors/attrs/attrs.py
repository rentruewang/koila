# Copyright (c) AIoWay Authors - All Rights Reserved

"Schema is a collection of metadata describing the 'type' of data."

import dataclasses as dcls
import typing
from collections import abc as cabc

import torch

from aioway import _tracking
from aioway._tracking import logging

from . import devices, dtypes, shapes

__all__ = ["Attr", "attr"]


LOGGER = logging.get_logger(__name__)
TRACKER = _tracking.get_tracker(lambda: Attr)


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


type AttrLike = Attr | AttrDict | torch.Tensor


def attr(item: AttrLike, /) -> Attr:
    "The convenient constructor function for `Attr` to convert from similar types."

    if isinstance(item, Attr):
        return item

    if isinstance(item, torch.Tensor):
        return Attr.from_tensor(item)

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
