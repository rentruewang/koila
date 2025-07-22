# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from abc import ABC
from collections.abc import Iterable, Iterator

from aioway import registries
from aioway.execs.execs import Exec
from aioway.execs.utils import Nargs

from .batches import Batch, Batch0, Batch1, Batch2

__all__ = [
    "Poller",
    "Poller0",
    "Poller1",
    "NextPoller",
    "Poller2",
    "ZipPoller",
    "NestedLoopPoller",
]


@dcls.dataclass(frozen=True)
class Poller(Nargs, Iterable[Batch], ABC):
    """
    ``Poller`` is responsible for polling data from the previous ``Exec``, if any.
    """

    def __init_subclass__(cls, *, key: str = ""):
        registries.init_subclass(lambda: Poller)(cls, key=key)

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Batch]: ...

    @property
    @abc.abstractmethod
    def children(self) -> Iterator[Exec]: ...


@dcls.dataclass(frozen=True)
class Poller0(Poller, key="SOURCE"):
    N_ARY = 0

    @typing.override
    def __iter__(self) -> Iterator[Batch0]:
        while True:
            yield Batch0()

    @property
    @typing.final
    def children(self):
        return
        yield


@dcls.dataclass(frozen=True)
class Poller1(Poller):
    N_ARY = 1

    child: Exec

    @typing.override
    @abc.abstractmethod
    def __iter__(self) -> Iterator[Batch1]: ...

    @property
    @typing.final
    def children(self):
        yield self.child


class NextPoller(Poller1, key="NEXT"):
    N_ARY = 1

    @typing.override
    def __iter__(self):
        for item in self.child:
            yield Batch1(item)


@dcls.dataclass(frozen=True)
class Poller2(Poller):
    N_ARY = 2

    left: Exec
    right: Exec

    @typing.override
    @abc.abstractmethod
    def __iter__(self) -> Iterator[Batch2]: ...

    @property
    @typing.final
    def children(self):
        yield self.left
        yield self.right


class ZipPoller(Poller2, key="ZIP"):
    N_ARY = 2

    @typing.override
    def __iter__(self):
        for l, r in zip(self.left, self.right):
            yield Batch2(l, r)


class NestedLoopPoller(Poller2, key="NESTED_LOOP"):
    N_ARY = 2

    @typing.override
    def __iter__(self):
        for l in self.left:
            for r in self.right:
                yield Batch2(l, r)
