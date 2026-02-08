# Copyright (c) AIoWay Authors - All Rights Reserved

"Schema is a collection of metadata describing the 'type' of data."

import copy
import dataclasses as dcls
import logging
import typing
from collections.abc import Iterator, Mapping
from typing import Self

from .devices import Device
from .dtypes import DType
from .shapes import Shape

__all__ = ["Attr", "AttrSet"]

LOGGER = logging.getLogger(__name__)


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


@dcls.dataclass(frozen=True)
class AttrSet(Mapping[str, Attr]):
    """
    A set of attributes. Representing the schema in a ``Table``.
    """

    attrs: dict[str, Attr] = dcls.field(default_factory=dict)
    """
    The data backing the ``AttrSet``.
    """

    @typing.override
    def __repr__(self) -> str:
        return repr(self.attrs)

    @typing.override
    def __len__(self) -> int:
        return len(self.attrs)

    @typing.override
    def __getitem__(self, key: str) -> Attr:
        return self.attrs[key]

    @typing.override
    def __iter__(self) -> Iterator[str]:
        return iter(self.attrs)

    def to_dict(self) -> dict[str, Attr]:
        return copy.deepcopy(self.attrs)

    @classmethod
    def init(cls, **attrs: Attr) -> Self:
        return cls(attrs)
