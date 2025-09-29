# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import functools
import logging
import typing
from collections.abc import Iterator
from typing import NamedTuple, Self

from tensordict import TensorDict

from aioway.intakes import TorchListFrame
from aioway.thunks import Thunk

from .execs import Exec

__all__ = ["CacheExec"]

LOGGER = logging.getLogger(__name__)


class CacheExec(Exec):
    """
    ``CacheExec`` caches the input,
    using a proxy ``Frame`` to store the intermediate result.
    """

    def __init__(self, exe: Exec) -> None:
        super().__init__()
        self._exec: Exec = exe

    @typing.override
    def __repr__(self) -> str:
        return f"Cache({self._exec})"

    @typing.override
    def iter(self):
        """
        Every ``__iter__`` call, create a new ``Generator`` that shares state.
        """

        # This should return the same instance over the lifetime of the ``CacheExec``.
        yield from self._cached_gen

    @functools.cached_property
    def _cached_gen(self):
        def iterator():
            "Iterator that keeps track of the indices."
            for idx, batch in enumerate(self._exec):
                yield BatchIdx(idx=idx, batch=batch)

        return _CachedThunkGen(iterator(), self.frame)

    @typing.override
    def inputs(self):
        yield self._exec

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
    iterator: Iterator[BatchIdx]
    frame: TorchListFrame
    idx: int = dcls.field(init=False, default=0)

    @typing.override
    def __iter__(self) -> Self:
        return self

    @typing.override
    def __next__(self) -> TensorDict:
        batch = self._compute_next()
        self.idx += 1
        return batch

    def _compute_next(self) -> TensorDict:
        assert self.idx >= 0
        assert self.idx <= len(self.frame)

        # Hitting unvisited territory.
        if self.idx == len(self.frame):
            next_idx, batch = next(self.iterator)
            assert next_idx == self.idx, {"self.idx": self.idx, "next_idx": next_idx}
            self.frame.append(batch)
        # Or else it must be seen before, so we can access it directly.
        else:
            batch = self.frame[self.idx]
        return batch
