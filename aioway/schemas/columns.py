# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
from typing import NamedTuple, Self

from torch import Tensor

from .devices import Device
from .dtypes import DType
from .shapes import Shape

__all__ = ["ColumnSchema"]


@dcls.dataclass(frozen=True)
class ColumnSchema:
    """
    ``ColumnSchema`` refers to the schema a column has.
    """

    dtype: DType | None = None
    """
    The data type for individual element.
    """

    shape: Shape | None = None
    """
    The shape of the tensor in this column, per element.
    """

    device: Device | None = None
    """
    The device on which the data would be transfered over for computation.
    """

    @classmethod
    def parse(cls, *, dtype, shape, device) -> Self:
        return cls(
            dtype=DType.parse(dtype),
            shape=Shape(tuple(map(int, shape))),
            device=Device.parse(device),
        )

    @classmethod
    def parse_tensor(cls, tensor: Tensor, /) -> Self:
        return cls.parse(dtype=tensor.dtype, shape=tensor.shape, device=tensor.device)


class NamedColumnSchema(NamedTuple):
    name: str
    """
    The name of the column.
    """

    schema: ColumnSchema
    """
    The schema for the column.
    """
