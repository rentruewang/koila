# Copyright (c) RenChu Wang - All Rights Reserved

import abc
from abc import ABC
from collections.abc import Iterable, Iterator

from aioway.blocks import Block
from aioway.datatypes import AttrSet

__all__ = ["Exec"]


class Exec(Iterator[Block], Iterable[Block], ABC):
    """
    ``Exec`` represents a stream of heterogenious data being generated,
    it is one of the main physical abstractions in ``aioway`` to represent eager computation.

    ``Exec`` acts like a generator, where it is an ``Iterator`` and ``Iterable``,
    so it can be used in for loops and yield expressions easily.

    It can be thought of as an ``Iterator`` of ``Block``s,
    where computation happens eagerly, imperatively, and the result is yielded.

    Todo:
        Since ``Exec`` itself is an ``Iterator``,
        some ``Iterable`` would be needed to handle initialization of ``Execs``.
    """

    @abc.abstractmethod
    def __next__(self) -> Block:
        """
        ``Iterator`` to yield from the current ``Exec``.
        """

        ...

    @property
    @abc.abstractmethod
    def attrs(self) -> AttrSet:
        """
        The output schema of the current table.
        """

        ...
