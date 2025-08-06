# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import logging
import typing
from collections.abc import Mapping
from typing import Any, Protocol, Self, TypedDict

from aioway.attrs.devices import Device
from aioway.attrs.dtypes import DType
from aioway.attrs.shapes import Shape

from .attrs import Attr, AttrDict, AttrInitTypeError, AttrObj

__all__ = ["NamedAttr"]

LOGGER = logging.getLogger(__name__)


class NamedAttrDict(AttrDict, TypedDict):
    name: Any


@typing.runtime_checkable
class NamedAttrObj(AttrObj, Protocol):
    @property
    def name(self) -> Any: ...


@dcls.dataclass(frozen=True)
class NamedAttr(Attr):
    _: dcls.KW_ONLY
    """
    Only allow keyword variables to prevent confusion.
    """

    name: str
    """
    The name of the column.
    """

    @classmethod
    def __init(cls, *, dtype, shape, device, name) -> Self:
        return cls(
            dtype=DType.parse(dtype),
            shape=Shape.from_iterable(shape),
            device=Device.parse(device),
            name=str(name),
        )

    @classmethod
    def parse(cls, like: Any) -> Self:
        logging.debug("Parsing %s", like)

        if isinstance(like, NamedAttrObj):
            return cls.__init(
                device=like.device, dtype=like.dtype, shape=like.shape, name=like.name
            )

        if isinstance(like, Mapping) and all(
            key in like.keys() for key in AttrDict.__annotations__
        ):
            return cls.__init(
                device=like["device"],
                dtype=like["dtype"],
                shape=like["shape"],
                name=like["name"],
            )

        raise NamedAttrInitTypeError(f"Cannot initialize non-attr-like {like}.")


class NamedAttrInitTypeError(AttrInitTypeError): ...
