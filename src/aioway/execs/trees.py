# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import logging
import typing
from collections.abc import Generator
from typing import Self

from aioway.ops import BatchGen

from .execs import Exec

__all__ = ["TreeExec"]

LOGGER = logging.getLogger(__name__)


@dcls.dataclass(frozen=True)
class TreeExec(Exec, key="TREE"):
    """
    ``LazyExec`` is the simplest implementation of ``Exec``,
    iterates over the graph with lazy evaluation.
    """

    @typing.override
    def __iter__(self) -> BatchGen:
        """
        Yields a generator, locally, from ``Op``'s definition.

        Returns:
            A stream of ``Block``s.

        Note:
            Always creates a new ``Generator`` upon being called, not cached.
        """

        yield from self.thunk.op.apply(*self._input_execs())

    def _input_execs(self) -> Generator[Self]:
        yield from (self.from_thunk(ipt) for ipt in self.thunk.inputs)
