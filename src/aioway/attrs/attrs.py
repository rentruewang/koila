# Copyright (c) AIoWay Authors - All Rights Reserved

"Schema is a collection of metadata describing the 'type' of data."

import dataclasses as dcls

from aioway.tables import Column

from . import devices, dtypes, shapes
from .devices import Device, DeviceLike
from .dtypes import DType, DTypeLike
from .shapes import Shape, ShapeLike

__all__ = ["Attr", "attr"]


@dcls.dataclass(frozen=True)
class Attr(Column):
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


def attr(device: DeviceLike, dtype: DTypeLike, shape: ShapeLike) -> Attr:
    return Attr(
        device=devices.device(device),
        dtype=dtypes.dtype(dtype),
        shape=shapes.shape(shape),
    )
