# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import logging
import typing
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Self

from tensordict import TensorDict
from torch import Tensor

from aioway._errors import AiowayError
from aioway.attrs.devices import Device
from aioway.attrs.dtypes import DType
from aioway.attrs.shapes import Shape

from .attrs import Attr
from .names import NamedAttr

__all__ = ["AttrSet"]

LOGGER = logging.getLogger(__name__)


@dcls.dataclass(frozen=True)
class AttrSet(Mapping[str, Attr]):
    """
    ``AttrSet`` is a set of ``Attr``s, typically used to represent the a ``Block``'s data type.

    ``AttrSet`` shows the schema of an entire table, vs ``Attr``'s description of a column.

    An ``Attr`` can be used to initialize a ``AttrSet``,
    which is a mapping of string to ``Attr``.

    Note that ``Attr``s are not ``Schema``s, the latter is used for integrating with external DBs.

    The benefit of separating ``Attr``s and ``Schema``s is that
    we are able to ``encode`` data of schema to ``AttrSet`` schema,
    which splits the internal and external data format handling,
    allowing a wider range of support.

    """

    columns: dict[str, Attr] = dcls.field(default_factory=dict)
    """
    The names and the types associated with the columns.

    Even though any mapping should work, the mapping must be order preserving.
    Therefore, here, a ``dict`` is specified.
    """

    device: Device | None = None
    """
    The global device to use, if specified.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.columns, dict):
            raise AttrSetInitError(
                f"Columns data format should be tuple, got {type(self.columns)=}."
            )

        if self.device and not isinstance(self.device, Device):
            raise AttrSetInitError(
                f"The device should be `Device` type, got {type(self.device)=}"
            )

        return NotImplemented

    @typing.override
    def __iter__(self) -> Iterator[str]:
        yield from self.columns

    @typing.override
    def __len__(self) -> int:
        return len(self.columns)

    @typing.overload
    def __getitem__(self, key: str) -> Attr: ...

    @typing.overload
    def __getitem__(self, key: list[str]) -> Self: ...

    @typing.override
    def __getitem__(self, key):
        LOGGER.debug("Getting key: %s on: %s", key, self)

        if isinstance(key, str):
            return self.columns[key]

        if isinstance(key, list) and all(isinstance(k, str) for k in key):
            return type(self)(columns={k: self[k] for k in key}, device=self.device)

        raise AttrGetKeyError(
            "Key type mismatch. Must be a string or a list of strings."
        )

    @typing.override
    def __contains__(self, key: object) -> bool:
        return key in self.columns

    @typing.override
    def __eq__(self, other: object):
        LOGGER.debug("Comparing %s == %s", lambda: self, lambda: other)

        if isinstance(other, AttrSet):
            return self.columns == other.columns

        # Do not check devices if RHS is mapping.
        if isinstance(other, Mapping):
            return self.columns == other

        return NotImplemented

    def __or__(self, other: Self) -> Self:
        LOGGER.debug("Computing %s | %s", self, other)

        # Using the logic in `__and__` to verify intersection.
        _ = self & other

        return type(self)({**self.columns, **other.columns}, device=self.device)

    def __and__(self, other: Self) -> Self:
        LOGGER.debug("Computing %s & %s", self, other)

        joint = set(self.keys()).intersection(other.keys())

        if not all(self[key] == other[key] for key in joint):
            raise AttrMergeError(
                f"Schema {self} and {other} has different dtypes on intersecting keys."
            )

        # If global device is specified, and not equal then we cannot merge.
        if self.device and other.device and self.device != other.device:
            raise AttrMergeError("Device is not equal.")

        return type(self)(
            {key: col for key, col in self.columns.items() if key in joint},
            device=self.device,
        )

    def product(self, other: Self, on: str) -> Self:
        if on not in self:
            raise AttrSetKeyError(f"{self.columns=} must contain key={on}")

        if on not in other:
            raise AttrSetKeyError(f"{other.columns=} must contain key={on}")

        # Merging here is OK, as `dict` update overwrites the left side.
        return self | other

    def project(self, *columns: str) -> Self:
        if not all(col in self for col in columns):
            raise AttrSetKeyError(f"Schema {self} does not contain all {columns=}")

        return type(self)({col: self[col] for col in columns}, device=self.device)

    def rename(self, **mapping: str) -> Self:
        if not all(key in self for key in mapping):
            raise AttrSetKeyError(f"{self} must be a superset of {list(mapping)}")

        return type(self)(
            {mapping.get(col, col): self[col] for col in {*mapping, *self}}
        )

    def transform(self, target: Self) -> Self:
        return target

    def union(self, other: Self) -> Self:
        if self != other:
            raise AttrMergeError(f"In union, {self} != {other}.")

        return self

    @property
    def names(self) -> list[str]:
        return list(self.columns.keys())

    @property
    def shapes(self) -> list[Shape]:
        return self._attr(lambda attr: attr.shape)

    @property
    def devices(self) -> list[Device]:
        return self._attr(lambda attr: attr.device)

    @property
    def dtypes(self) -> list[DType]:
        return self._attr(lambda attr: attr.dtype)

    def _attr[T](self, getter: Callable[[Attr], T]) -> list[T]:
        return [getter(val) for val in self.columns.values()]

    @classmethod
    def from_iterable(
        cls, columns: Iterable[NamedAttr], device: Device = Device()
    ) -> Self:
        """
        Creates an ``AttrSet`` object from an iterable of ``AttrWithName`` objects.
        """

        LOGGER.debug("Creating attribute set from iterable. %s", columns)

        return cls(
            {
                col.name: Attr(dtype=col.dtype, shape=col.shape, device=col.device)
                for col in columns
            },
            device=device,
        )

    @classmethod
    def parse_tensor_dict(
        cls, td: dict[str, Tensor] | TensorDict, device: str | None = None
    ) -> Self:
        """
        Creates an ``AttrSet`` object from a dictionary of tensors.
        """

        td = TensorDict(td)

        LOGGER.debug(
            "Creating attribute set from dict, dtype=%s, shape=%s, device=%s",
            td.dtype,
            td.shape,
            td.device,
        )

        return cls(
            {key: Attr.parse_tensor(tensor) for key, tensor in td.items()},
            device=Device.parse(device),
        )


class AttrSetInitError(AiowayError, AssertionError, TypeError): ...


class AttrMergeError(AiowayError, KeyError): ...


class AttrGetKeyError(AiowayError, KeyError): ...


class AttrSetKeyError(AiowayError, KeyError): ...
