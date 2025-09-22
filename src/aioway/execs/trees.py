# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from aioway.ops import BatchGen, Thunk

from .execs import Exec, ExecCtx

__all__ = ["TreeExec"]


class TreeExec(Exec, key="TREE"):
    """
    ``TreeExec`` is the simplest implementation of ``Exec``,
    iterates over the graph with lazy evaluation.
    """

    def __init__(self, thunk: Thunk, /, *ctxs: ExecCtx) -> None:
        super().__init__(*ctxs)
        self._thunk = thunk

    @typing.override
    def iterate(self) -> BatchGen:
        """
        Yields a generator, locally, from ``Op``'s definition.

        Returns:
            A stream of ``Block``s.

        Note:
            Always creates a new ``Generator`` upon being called, not cached.
        """

        yield from self._thunk.op(*self.inputs())

    @typing.override
    def inputs(self):
        for ipt in self._thunk.inputs():
            yield type(self)(ipt)

    @property
    @typing.override
    def thunk(self):
        return self._thunk
