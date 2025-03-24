# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
from typing import NamedTuple, Self

from torch import Tensor

from aioway.errors import AiowayError

from .devices import Device
from .dtypes import DType
from .shapes import Shape

__all__ = ["Attr", "AttrWithName"]


@dcls.dataclass(frozen=True)
class Attr:
    """
    ``Attr`` refers to the attributes a column uses.
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

    def __post_init__(self) -> None:
        if self.dtype is None:
            raise AttrNullMemberError("`Attr.dtype` cannot be `None`.")

        if self.shape is None:
            raise AttrNullMemberError("`Attr.shape` cannot be `None`.")

        if self.device is None:
            raise AttrNullMemberError("`Attr.device` cannot be `None`.")

    @classmethod
    def parse(cls, *, dtype, shape, device) -> Self:
        return cls(
            dtype=DType.parse(dtype),
            shape=Shape.from_seq(shape),
            device=Device.parse(device),
        )

    @classmethod
    def parse_tensor(cls, tensor: Tensor, /, discard_batch_dim: bool = True) -> Self:
        """
        Parse the current ``Attr`` instance from a ``Tensor``.

        Args:
            tensor: The tensor for which to create an attribute for.
            discard_batch_dim:
                Discard the first dimension of the tensor's shape.
                Useful when initializing an ``AttrSet`` because ``shape`` in ``AttrSet`` dicards the batch size.
        """

        shape = tensor.shape[1:] if discard_batch_dim else tensor.shape

        return cls.parse(dtype=tensor.dtype, shape=shape, device=tensor.device)


class AttrWithName(NamedTuple):
    name: str
    """
    The name of the column.
    """

    attr: Attr
    """
    The schema for the column.
    """


class AttrNullMemberError(AiowayError, ValueError): ...
