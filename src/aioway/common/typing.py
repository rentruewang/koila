# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from collections.abc import Iterable, Iterator
from typing import Generic, Protocol, TypeVar

_T = TypeVar("_T")
_K = TypeVar("_K", contravariant=True)
_V = TypeVar("_V", covariant=True)


@typing.runtime_checkable
class Len(Protocol):
    @abc.abstractmethod
    def __len__(self) -> int: ...


@typing.runtime_checkable
class GetItem(Protocol[_K, _V]):
    @abc.abstractmethod
    def __getitem__(self, key: _K) -> _V: ...


@typing.runtime_checkable
class Seq(Len, GetItem[int, _V], Protocol[_V]): ...


@typing.runtime_checkable
class Array(Len, GetItem[int | slice, _V], Protocol[_V]): ...


@typing.runtime_checkable
class KeysAndGetItem(GetItem[_T, _V], Protocol[_T, _V]):
    """
    ``KeysAndGetItem`` is temporary fill in before the interface
    is officially part of the public standard library API.
    """

    @abc.abstractmethod
    def keys(self) -> Iterable[_T]: ...


@dcls.dataclass(frozen=True)
class ValuesView(Iterable[_V], Generic[_K, _V]):
    mapping: KeysAndGetItem[_K, _V]

    def __post_init__(self) -> None:
        if not isinstance(self.mapping, KeysAndGetItem):
            raise ValueError(
                "`mapping` should support both `__getitem__(idx)` and `keys()`."
            )

    def __iter__(self) -> Iterator[_V]:
        for key in self.mapping.keys():
            yield self.mapping[key]


@dcls.dataclass(frozen=True)
class ItemsView(Iterable[tuple[_K, _V]]):
    mapping: KeysAndGetItem[_K, _V]

    def __post_init__(self) -> None:
        if not isinstance(self.mapping, KeysAndGetItem):
            raise ValueError(
                "`mapping` should support both `__getitem__(idx)` and `keys()`."
            )

    def __iter__(self) -> Iterator[tuple[_K, _V]]:
        for key in self.mapping.keys():
            yield key, self.mapping[key]
