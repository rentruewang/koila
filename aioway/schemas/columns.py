# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
from typing import NamedTuple

from .devices import Device
from .dtypes import DType
from .shapes import Shape

__all__ = ["ColumnSchema"]


@dcls.dataclass(frozen=True)
class ColumnSchema:
    """
    ``ColumnSchema`` refers to the schema a column has.
    """

    dtype: DType
    """
    The data type for individual element.
    """

    shape: Shape
    """
    The shape of the tensor in this column, per element.
    """

    device: Device
    """
    The device on which the data would be transfered over for computation.
    """


class NamedColumnSchema(NamedTuple):
    name: str
    """
    The name of the column.
    """

    schema: ColumnSchema
    """
    The schema for the column.
    """
