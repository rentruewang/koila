# Copyright (c) AIoWay Authors - All Rights Reserved


import abc
import dataclasses as dcls
import typing
from abc import ABC
from collections.abc import Iterator

from .batches import (
    Batch,
    BinaryBatch,
    NullaryBatch,
    UnaryBatch,
)
from .execs import Exec


@dcls.dataclass(frozen=True)
class Poller(ABC):
    """
    ``Poller`` is responsible for polling data from the previous ``Exec``, if any.
    """

    @abc.abstractmethod
    def __next__(self) -> Batch: ...

    @abc.abstractmethod
    def children(self) -> Iterator[Exec]: ...


@dcls.dataclass(frozen=True)
class NullaryPoller(ABC):
    @typing.final
    def __next__(self) -> NullaryBatch:
        return NullaryBatch()

    @typing.final
    def children(self):
        return
        yield


@dcls.dataclass(frozen=True)
class UnaryPoller(ABC):
    left: Exec

    @abc.abstractmethod
    def __next__(self) -> UnaryBatch: ...

    @typing.final
    def children(self):
        yield self.left


@dcls.dataclass(frozen=True)
class BinaryPoller(ABC):
    left: Exec
    right: Exec

    @abc.abstractmethod
    def __next__(self) -> BinaryBatch: ...

    @typing.final
    def children(self):
        yield self.left
        yield self.right
