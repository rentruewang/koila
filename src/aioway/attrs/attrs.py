# Copyright (c) AIoWay Authors - All Rights Reserved

"Schema is a collection of metadata describing the 'type' of data."

import dataclasses as dcls

from .devices import Device
from .dtypes import DType
from .shapes import Shape

__all__ = ["Attr"]


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
