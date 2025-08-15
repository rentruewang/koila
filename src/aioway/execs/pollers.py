# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from abc import ABC
from collections.abc import Iterable, Iterator
from typing import Self

from aioway.errors import AiowayError

from .batches import Batch, Batch0, Batch1, Batch2
from .execs import Exec
from .utils import Nargs

__all__ = [
    "Poller",
    "Poller0",
    "Poller1",
    "NoopPoller",
    "RepeatPoller",
    "Poller2",
    "ZipPoller",
    "NestedLoopPoller",
]


@dcls.dataclass(frozen=True)
class Poller[B: Batch](Nargs, Iterable[B], ABC):
    """
    ``Poller`` is responsible for polling data from the previous ``Exec``, if any.
    """

    def __init_subclass__(cls, *, key: str = ""):
        cls._init_subclass(Poller, key=key)

    def __iter__(self) -> Iterator[B]:
        for item in self._iterate():
            # Ensure that ``ARGC`` matches.

            if self.ARGC != item.ARGC:
                raise PollerArgcMismatch(f"{self.ARGC=} != {item.ARGC=}.")

            yield item

    @abc.abstractmethod
    def _iterate(self) -> Iterator[B]: ...

    @property
    @abc.abstractmethod
    def execs(self) -> tuple[Exec, ...]: ...

    @classmethod
    def init(cls, *execs: "Exec") -> Self:
        if len(execs) != cls.ARGC:
            raise PollerInitError(
                f"Poller {cls} only has {cls.ARGC} children. "
                f"Got {len(execs)} arguments: {execs}."
            )

        return cls(*execs)


@typing.final
@dcls.dataclass(frozen=True)
class Poller0(Poller[Batch0], key="NOOP_0"):
    ARGC = 0

    @typing.override
    def _iterate(self) -> Iterator[Batch0]:
        while True:
            yield Batch0()

    @property
    @typing.final
    def execs(self) -> tuple[()]:
        return ()


@dcls.dataclass(frozen=True)
class Poller1(Poller[Batch1], ABC):
    ARGC = 1

    child: "Exec"

    @typing.override
    @abc.abstractmethod
    def _iterate(self) -> Iterator[Batch1]: ...

    @property
    @typing.final
    def execs(self) -> tuple[Exec]:
        return (self.child,)


@typing.final
@dcls.dataclass(frozen=True)
class NoopPoller(Poller1, key="PASS_1"):
    ARGC = 1

    @typing.override
    def _iterate(self):
        for item in self.child:
            yield Batch1(item)


@typing.final
@dcls.dataclass(frozen=True)
class RepeatPoller(Poller1, key="REPEAT_1"):
    ARGC = 1

    repeat: int = 1

    @typing.override
    def _iterate(self):
        for item in self.child:
            for _ in range(self.repeat):
                yield Batch1(item)


@dcls.dataclass(frozen=True)
class Poller2(Poller[Batch2], ABC):
    ARGC = 2

    left: "Exec"
    right: "Exec"

    @typing.override
    @abc.abstractmethod
    def _iterate(self) -> Iterator[Batch2]: ...

    @property
    @typing.final
    def execs(self) -> tuple[Exec, Exec]:
        return self.left, self.right


@typing.final
@dcls.dataclass(frozen=True)
class ZipPoller(Poller2, key="ZIP_2"):
    ARGC = 2

    @typing.override
    def _iterate(self):
        for l, r in zip(self.left, self.right):
            yield Batch2(l, r)


@typing.final
@dcls.dataclass(frozen=True)
class NestedLoopPoller(Poller2, key="NESTED_2"):
    ARGC = 2

    @typing.override
    def _iterate(self):
        for l in self.left:
            for r in self.right:
                yield Batch2(l, r)


class PollerInitError(AiowayError, TypeError): ...


class PollerArgcMismatch(AiowayError, TypeError): ...
