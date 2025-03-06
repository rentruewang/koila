# Copyright (c) RenChu Wang - All Rights Reserved

import abc
from abc import ABC
from collections.abc import Iterable, Iterator
from typing import Self

from aioway.blocks import Block
from aioway.tables import Table

__all__ = ["Stream"]


class Stream(Table, Iterator[Block], Iterable[Block], ABC):
    """
    ``Stream`` represents a stream of heterogenious data being generated,
    it is one of the main physical abstractions in ``aioway`` to represent eager computation.

    ``Stream`` acts like a generator, where it is an ``Iterator`` and ``Iterable``,
    so it can be used in for loops and yield expressions easily.

    It can be thought of as an ``Iterator`` of ``Block``s,
    where computation happens eagerly, imperatively, and the result is yielded.

    Todo:
        Since ``Stream`` itself is an ``Iterator``,
        some ``Iterable`` would be needed to handle initialization of ``Streams``.
    """

    @abc.abstractmethod
    def __next__(self) -> Block:
        """
        ``Iterator`` to yield from the current ``Stream``.
        """

        ...

    def __length_hint__(self) -> int:
        """
        The estimated number of elements in the stream.
        Per PEP-424, does not need to be accurate.

        Returns:
            An integer, and must be an ``int``, not a subclass.
        """

        return NotImplemented
