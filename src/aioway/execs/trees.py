# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from aioway.ops import BatchGen, Thunk

from .execs import Exec

__all__ = ["TreeExec"]


class TreeExec(Exec, key="TREE"):
    """
    ``TreeExec`` is the simplest implementation of ``Exec``,
    iterates over the graph with lazy evaluation.
    """

    def __init__(self, thunk: Thunk, /) -> None:
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

        def input_execs():
            for ipt in self._thunk.inputs:
                yield type(self)(ipt)

        yield from self._thunk.op(*input_execs())

    @property
    @typing.override
    def thunk(self):
        return self._thunk
