# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import typing
from abc import ABC
from collections.abc import Iterator

from aioway.errors import AiowayError

__all__ = ["Factory", "Registry"]


@typing.final
@dcls.dataclass(frozen=True, slots=True)
class Factory[T: type[ABC]]:
    """
    The factory class.

    Use ``Factory.of`` to create a factory of another class.
    """

    lookup: dict[str, T] = dcls.field(default_factory=dict)
    """
    The per-type registry, storing the subclasses with their aliases as keys.
    """

    inverse: dict[T, str] = dcls.field(default_factory=dict)
    """
    The inverse of per-type registry, from subclasses to aliases.
    """

    base_class: T = object  # type: ignore[assignment]
    """
    The base class that must be the superclass of all values in the factory.
    """

    def __post_init__(self) -> None:
        for key, val in self.lookup.items():
            assert self.inverse[val] == key

        for val, key in self.inverse.items():
            assert self.lookup[key] == val

    def __iter__(self) -> Iterator[str]:
        yield from self.lookup.keys()

    def __contains__(self, key: str) -> bool:
        return key in self.lookup.keys()

    def __len__(self) -> int:
        return len(self.lookup)

    @typing.overload
    def __getitem__(self, key: str) -> T: ...

    @typing.overload
    def __getitem__(self, key: T) -> str: ...

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.lookup[key]

        if issubclass(key, self.base_class):
            return self.inverse[key]

        raise FactoryKeyError("Factory's key must either be string or key.")

    @typing.no_type_check
    def __setitem__(self, key: str, val: T) -> None:
        if not isinstance(key, str):
            raise FactoryKeyError(f"Key must be a string. Got {type(key)=}")

        if not issubclass(val, self.base_class):
            raise FactoryKeyError(f"Value must be a type. Got {type(val)=}")

        if key in self.lookup:
            existing = self.lookup[key]
            raise FactoryKeyError(f"Key: {key} already used for {existing}.")

        self.lookup[key] = val
        self.inverse[val] = key

    def __delitem__(self, key: str) -> None:
        val = self.lookup[key]
        del self.lookup[key]
        del self.inverse[val]


class FactoryKeyError(AiowayError, KeyError): ...
