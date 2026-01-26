# Copyright (c) AIoWay Authors - All Rights Reserved

"Schema is a collection of metadata describing the 'type' of data."

import dataclasses as dcls
import typing
from collections.abc import Iterator, Mapping

from .devices import Device
from .dtypes import DType
from .shapes import Shape


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


class AttrSet(Mapping[str, Attr]):
    """
    A set of attributes. Representing the schema in a ``Table``.
    """

    def __init__(self, **attrs: Attr):
        self._attrs: dict[str, Attr] = attrs

    @typing.override
    def __repr__(self) -> str:
        return repr(self._attrs)

    @typing.override
    def __len__(self) -> int:
        return len(self._attrs)

    @typing.override
    def __getitem__(self, key: str) -> Attr:
        return self._attrs[key]

    @typing.override
    def __iter__(self) -> Iterator[str]:
        return iter(self._attrs)
