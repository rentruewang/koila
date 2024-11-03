# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import abc
import typing
from collections.abc import Callable, Sequence
from typing import Protocol, TypeVar


@typing.runtime_checkable
class Equivalent(Protocol):
    @abc.abstractmethod
    def __eq__(self, other: object) -> bool: ...

    @abc.abstractmethod
    def __ne__(self, other: object) -> bool: ...


@typing.runtime_checkable
class Comparable(Equivalent, Protocol):
    @abc.abstractmethod
    def __lt__(self, other: object) -> bool: ...

    @abc.abstractmethod
    def __le__(self, other: object) -> bool: ...

    @abc.abstractmethod
    def __gt__(self, other: object) -> bool: ...

    @abc.abstractmethod
    def __ge__(self, other: object) -> bool: ...


_T = TypeVar("_T")
_Cmp = TypeVar("_Cmp", bound=Comparable)


def is_ordered(seq: Sequence[_T], /, key: Callable[[_T], _Cmp] | None = None) -> bool:
    func = key if key is not None else lambda x: typing.cast(_Cmp, x)
    return all(func(seq[i]) <= func(seq[i + 1]) for i in range(len(seq) - 1))
