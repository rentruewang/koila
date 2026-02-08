# Copyright (c) AIoWay Authors - All Rights Reserved

"Schema is a collection of metadata describing the 'type' of data."

import dataclasses as dcls
import functools
import typing
from collections.abc import Iterator, KeysView, Mapping, Sequence
from typing import NamedTuple, Self

import numpy as np

from . import attrs
from .attrs import Attr
from .devices import Device
from .dtypes import DType
from .shapes import Shape

__all__ = ["AttrSet"]


class AttrSetCol(NamedTuple):
    """
    The name and attribute for each column.
    """

    name: str
    "The name of the column."

    attr: Attr
    "The attribute that the column has."


@dcls.dataclass(frozen=True, repr=False)
class AttrSet(Mapping[str, Attr]):
    """
    A set of attributes. Representing the schema in a ``Table``.

    Right now the columns are in sorted order, but this is not guarenteed.
    Most likely will change in the future.
    """

    attrs: tuple[AttrSetCol, ...] = ()
    """
    The data backing the ``AttrSet``. Must be sorted.
    """

    def __post_init__(self) -> None:
        if len(self.names) > 1 and not np.all(self.names[:-1] <= self.names[1:]):
            raise ValueError(f"Names are not sorted: {self.names}.")

    @typing.override
    def __repr__(self) -> str:
        return self._repr_string

    @typing.override
    def __len__(self) -> int:
        return len(self.attrs)

    @typing.override
    def __getitem__(self, key: str) -> Attr:
        # Using the ``find`` function from ``AttrSetKeysView``, to be DRY.
        if (idx := self.keys().find(key)) is None:
            raise KeyError(key)

        assert 0 <= idx < len(self)
        return self.attrs[idx].attr

    @typing.override
    def __iter__(self) -> Iterator[str]:
        return (attr.name for attr in self.attrs)

    @typing.override
    def keys(self):
        return self._keys_view

    @functools.cached_property
    def _repr_string(self):
        kvs = (f"{k}:{v}" for k, v in self.attrs)
        return "{" + ", ".join(kvs) + "}"

    @functools.cached_property
    def _keys_view(self):
        return AttrSetKeysView(self)

    @functools.cached_property
    def names(self):
        return [col.name for col in self.attrs]

    @functools.cached_property
    def dtypes(self):
        return [col.attr.dtype for col in self.attrs]

    @functools.cached_property
    def shapes(self):
        return [col.attr.shape for col in self.attrs]

    @functools.cached_property
    def devices(self):
        return [col.attr.device for col in self.attrs]

    @classmethod
    def from_values(cls, **attrs: Attr) -> Self:
        return cls.from_dict(attrs)

    @classmethod
    def from_dict(cls, attrs: dict[str, Attr]) -> Self:
        return cls(
            tuple(
                sorted(AttrSetCol(name=name, attr=attr) for name, attr in attrs.items())
            )
        )

    @classmethod
    def from_fields(
        cls,
        *,
        names: Sequence[str],
        shapes: Sequence[Shape],
        dtypes: Sequence[DType],
        devices: Sequence[Device],
    ) -> Self:
        if not (len(names) == len(shapes) == len(dtypes) == len(devices)):
            raise ValueError(
                "Should all have the same length. "
                f"Got {len(names)=}, {len(shapes)=}, {len(dtypes)=}, {len(devices)=}."
            )

        mapping = {
            name: attrs.attr(device=device, dtype=dtype, shape=shape)
            for name, device, dtype, shape in zip(names, devices, dtypes, shapes)
        }

        return cls.from_dict(mapping)


@dcls.dataclass(frozen=True)
class AttrSetKeysView(KeysView[str]):
    attrset: AttrSet
    "The data to view. Its names must be sorted."

    @typing.override
    def __len__(self):
        return len(self.keys)

    @typing.override
    def __iter__(self):
        return iter(self.keys)

    @typing.override
    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self.find(key) is not None

    @typing.override
    @typing.no_type_check
    def __eq__(self, rhs: object) -> bool:
        try:
            rhs_set = set(rhs)

        # If rhs does not have ``__iter__``, ``TypeError`` would be raised.
        except TypeError:
            return False

        return self._set == rhs_set

    @functools.cached_property
    def _set(self):
        return set(self)

    def find(self, key: str) -> int | None:
        """
        Search the ``key`` in the keys.
        If found, return the index. If not found, return None.
        """

        idx = int(np.searchsorted(self.keys, key))

        if idx < len(self) and self.keys[idx] == key:
            return idx

        return None

    @property
    def keys(self) -> list[str]:
        return self.attrset.names
