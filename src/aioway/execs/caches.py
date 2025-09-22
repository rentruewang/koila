# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import functools
import logging
import typing
from collections.abc import Iterator
from typing import NamedTuple, Self

from tensordict import TensorDict

from aioway.io import TorchListFrame
from aioway.ops import Thunk

from .execs import Exec

__all__ = ["CacheExec"]

LOGGER = logging.getLogger(__name__)


class CacheExec(Exec, key="CACHE"):
    """
    ``LazyExec`` is the simplest implementation of ``Exec``,
    iterates over the graph with lazy evaluation.
    """

    def __init__(self, executor: Exec, /) -> None:
        self._exec = executor

    @typing.override
    def iterate(self):
        """
        Every ``__iter__`` call, create a new ``Generator`` that shares state.
        """

        generator = _CachedThunkGen(self)

        return generator

    @property
    @typing.override
    def thunk(self) -> Thunk:
        return self._exec.thunk

    @functools.cached_property
    def frame(self):
        return TorchListFrame()


class BatchIdx(NamedTuple):
    idx: int
    batch: TensorDict


@dcls.dataclass
class _CachedThunkGen(Iterator[TensorDict]):
    executor: CacheExec
    idx: int = dcls.field(init=False, default=0)

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> TensorDict:
        batch = self.compute()
        self.idx += 1
        return batch

    def compute(self) -> TensorDict:
        assert self.idx >= 0
        assert self.idx <= len(self.frame)

        if self.idx == len(self.frame):
            next_idx, batch = next(self.iterator)
            assert next_idx == self.idx
            self.frame.append(batch)
        else:
            batch = self.frame[self.idx]
        return batch

    @property
    def frame(self):
        return self.executor.frame

    @functools.cached_property
    def iterator(self):
        "Iterator that keeps track of the indices."
        for idx, batch in self.executor:
            yield BatchIdx(idx=idx, batch=batch)
