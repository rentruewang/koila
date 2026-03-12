# Copyright (c) AIoWay Authors - All Rights Reserved

"Schema is a collection of metadata describing the 'type' of data."

import dataclasses as dcls
import functools
import typing
from collections.abc import Iterator, KeysView, Mapping, Sequence
from typing import NamedTuple, Self

import numpy as np
from numpy import ndarray as NpArr
from torch import Tensor

from aioway import _typing
from aioway._tables import Table

from . import attrs, devices, dtypes, shapes
from .attrs import Attr
from .devices import Device, DeviceLike
from .dtypes import DType, DTypeLike
from .shapes import Shape, ShapeLike

__all__ = ["AttrSet", "DTypeSet", "DeviceSet", "ShapeSet", "AttrSetLike", "attr_set"]

type AttrSetLike = AttrSet | dict[str, Attr]


class _AttrItem[T](NamedTuple):
    """
    The name and attribute for each column.
    """

    name: str
    "The name of the column."

    attr: T
    "The attribute that the column has."


@dcls.dataclass(frozen=True, repr=False)
class _AttrSetBase[T](Table[T], Mapping[str, T]):
    """
    A set of attributes. Representing the schema in a ``Table``.

    Right now the columns are in sorted order, but this is not guarenteed.
    Most likely will change in the future.
    """

    attrs: tuple[_AttrItem[T], ...] = ()
    """
    The data backing the ``AttrSet``. Must be sorted.
    """

    def __post_init__(self) -> None:
        if len(self.names) > 1 and not np.all(self.names[:-1] <= self.names[1:]):
            raise ValueError(f"Names are not sorted: {self.names}.")

    @typing.override
    def __repr__(self) -> str:
        return self._repr_string

    def __or__(self, other: Mapping[str, T]) -> Self:
        return type(self).from_dict({**self, **other})

    @typing.override
    def __len__(self) -> int:
        return len(self.attrs)

    @typing.override
    def __iter__(self) -> Iterator[str]:
        return (attr.name for attr in self.attrs)

    @typing.override
    def keys(self):
        return self._keys_view

    @typing.override
    def column(self, key: str, /) -> T:
        # Using the ``find`` function from ``AttrSetKeysView``, to be DRY.
        if (idx := self.keys().find(key)) is None:
            raise KeyError(key)

        assert 0 <= idx < len(self)
        return self.attrs[idx].attr

    @typing.override
    def select(self, *keys: str) -> Self:
        return type(self).from_dict({key: self[key] for key in keys})

    @functools.cached_property
    def _repr_string(self):
        kvs = (f"{k}:{v}" for k, v in self.attrs)
        return "{" + ", ".join(kvs) + "}"

    @functools.cached_property
    def _keys_view(self):
        return _AttrKeysView(self)

    @functools.cached_property
    def names(self):
        return [col.name for col in self.attrs]

    @classmethod
    def from_values(cls, **attrs: T) -> Self:
        return cls.from_dict(attrs)

    @classmethod
    def from_dict(cls, attrs: Mapping[str, T], /) -> Self:
        return cls(
            tuple(
                sorted(_AttrItem(name=name, attr=attr) for name, attr in attrs.items())
            )
        )


class DTypeSet(_AttrSetBase[DType]):
    @classmethod
    @typing.override
    def from_dict(cls, attrs: Mapping[str, DTypeLike], /) -> Self:
        converted = {key: dtypes.dtype(dt) for key, dt in attrs.items()}
        return super().from_dict(converted)


class DeviceSet(_AttrSetBase[Device]):
    @classmethod
    @typing.override
    def from_dict(cls, attrs: Mapping[str, DeviceLike]) -> Self:
        converted = {key: devices.device(dev) for key, dev in attrs.items()}
        return super().from_dict(converted)


class ShapeSet(_AttrSetBase[Shape]):
    @classmethod
    @typing.override
    def from_dict(cls, attrs: Mapping[str, ShapeLike]) -> Self:
        converted = {key: shapes.shape(dev) for key, dev in attrs.items()}
        return super().from_dict(converted)


class AttrSet(_AttrSetBase[Attr]):

    @typing.overload
    def __getitem__(self, idx: str) -> Attr: ...

    @typing.overload
    def __getitem__(
        self, idx: int | slice | list[int] | list[str] | NpArr | Tensor
    ) -> Self: ...

    def __getitem__(self, idx):

        if isinstance(idx, str) or (
            isinstance(idx, list) and all(isinstance(i, str) for i in idx)
        ):
            return super().__getitem__(idx)

        names = self.names
        devices = self.devices
        shapes = self.shapes
        dtypes = self.dtypes

        if isinstance(idx, int):
            modified = [shape[1:] for shape in shapes]

        elif isinstance(idx, slice | NpArr | Tensor):
            modified = shapes[:]

        elif isinstance(idx, list) and all(isinstance(i, int) for i in idx):
            modified = shapes[:]

        else:
            raise TypeError(type(idx))

        return self.from_fields(
            names=names, devices=devices, dtypes=dtypes, shapes=modified
        )

    def rename(self, **renames: str):
        return self.from_dict(
            {renames.get(key, key): attr for key, attr in self.items()}
        )

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
    def from_fields(
        cls,
        *,
        names: Sequence[str],
        shapes: Sequence[Shape],
        dtypes: Sequence[DType],
        devices: Sequence[Device],
    ) -> Self:
        "Create an ``AttrSet`` from a set of seuqences of attributes of same length."

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

    @classmethod
    def from_sets(
        cls, *, shapes: ShapeSet, dtypes: DTypeSet, devices: DeviceSet
    ) -> Self:
        "Create an ``AttrSet`` from ``*Set`` types. Keys should match."

        shapes_keys = shapes.keys()
        dtypes_keys = dtypes.keys()
        devices_keys = devices.keys()

        if not (shapes_keys == dtypes_keys == devices_keys):
            raise ValueError(
                "All sets should have the same keys. "
                f"Got shapes={shapes_keys}, devices={devices_keys}, dtypes={dtypes_keys}"
            )

        names_list = list(shapes_keys)
        dtypes_list = [dtypes[key] for key in names_list]
        devices_list = [devices[key] for key in names_list]
        shapes_list = [shapes[key] for key in names_list]

        return cls.from_fields(
            names=names_list,
            dtypes=dtypes_list,
            devices=devices_list,
            shapes=shapes_list,
        )


@dcls.dataclass(frozen=True)
class _AttrKeysView(KeysView[str]):
    attrset: _AttrSetBase
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


def attr_set(schema: AttrSetLike, /) -> AttrSet:
    if isinstance(schema, AttrSet):
        return schema

    _is_dict_of_attr = _typing.is_dict_of_str_to(Attr)
    if _is_dict_of_attr(schema):
        return AttrSet.from_dict(schema)

    raise TypeError(type(schema))
