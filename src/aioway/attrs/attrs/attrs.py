# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import logging
import typing
from collections.abc import Mapping
from typing import Any, Protocol, Self, TypedDict

from torch import Tensor

from aioway._errors import AiowayError
from aioway.attrs.devices import Device
from aioway.attrs.dtypes import DType
from aioway.attrs.shapes import Shape

__all__ = ["Attr"]

LOGGER = logging.getLogger(__name__)


class AttrDict(TypedDict):
    dtype: Any
    shape: Any
    device: Any


@typing.runtime_checkable
class AttrObj(Protocol):
    @property
    def dtype(self) -> Any: ...

    @property
    def shape(self) -> Any: ...

    @property
    def device(self) -> Any: ...


@dcls.dataclass(frozen=True)
class Attr:
    """
    ``Attr`` refers to the attributes a column uses.
    """

    _: dcls.KW_ONLY
    """
    Only allow keyword variables to prevent confusion.
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
    def __init(cls, *, dtype, shape, device) -> Self:
        return cls(
            dtype=DType.parse(dtype),
            shape=Shape.from_iterable(shape),
            device=Device.parse(device),
        )

    @classmethod
    def parse(cls, like: Any) -> Self:
        LOGGER.debug("Parsing: %s", like)

        if isinstance(like, AttrObj):
            return cls.__init(
                device=like.device,
                dtype=like.dtype,
                shape=like.shape,
            )

        if isinstance(like, Mapping) and all(
            key in like.keys() for key in AttrDict.__annotations__
        ):
            return cls.__init(
                device=like["device"],
                dtype=like["dtype"],
                shape=like["shape"],
            )

        raise AttrInitTypeError(f"Cannot initialize non-attr-like {like}.")

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

        return cls.__init(dtype=tensor.dtype, shape=shape, device=tensor.device)


class AttrNullMemberError(AiowayError, ValueError): ...


class AttrInitTypeError(AiowayError, TypeError): ...
