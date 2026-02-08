# Copyright (c) AIoWay Authors - All Rights Reserved

"The ``Stream`` interfaces live here."

import abc
import dataclasses as dcls
import functools
import typing
from abc import ABC
from collections.abc import Generator, Iterator
from typing import ClassVar, Self

from aioway.batches import Chunk

__all__ = ["Stream"]


@dcls.dataclass
class Stream(Iterator[Chunk], ABC):
    """
    ``Stream`` produces a stream of batches of data, in the form of ``TensorDict``s,
    everytime ``__next__`` is called on it, a ``TensorDict`` is yielded.

    ``Stream`` is a stateful operation, compared to the previous implementations,
    it is an external iterator, supporting state inspection, simplifying debugging.
    """

    __match_args__: ClassVar[tuple[str, ...]]
    """
    A ``Stream`` should be able to be decomposed with ``match`` statements.
    """

    @typing.override
    def __iter__(self) -> Self:
        """
        ``__iter__`` allows ``Stream``s to be used in ``for`` loops.

        As it returns ``self``, re-use carefully.
        """

        return self

    @typing.final
    @typing.override
    def __next__(self) -> Chunk:
        """
        ``__next__`` allows ``Stream``s to be used in ``for`` loops.
        """

        result = self._read()
        self.__done_count += 1
        return result

    @property
    @abc.abstractmethod
    def size(self) -> int:
        """
        The length of the current ``Stream``.
        Does not change when the ``Stream`` is being iterated over.
        """

        ...

    @abc.abstractmethod
    def _read(self) -> Chunk:
        """
        Compute the next batch (a ``Chunk``).

        An exception raised here would be translated to ``StopIteration``.

        Note:

            The name ``read`` is inspired by the original release of clickhouse.
            I think this is a great name for a as a method on ``Stream`` to poll data,
            giving it a natural feeling like how we normally work with files.

        """

        ...

    @typing.final
    def children(self) -> Iterator["Stream"]:
        """
        Yields the children (dependencies of the current ``Stream``).

        Does not yield ``self``.

        Yields:
            Some ``Stream``s that would be evaluated if the current is running.
        """
        for child in self._children():
            assert child is not self, "A `Stream` cannot depend on itself!"
            yield child

    @abc.abstractmethod
    def _children(self) -> Generator["Stream"]:
        """
        ``Stream``'s children, the dependent ``Stream``s that would also be evaluated
        when calling ``__next__`` on the current ``Stream``.
        """

        ...

    @property
    def idx(self) -> int:
        """
        The number of batches completed..
        """

        return self.__done_count

    @functools.cached_property
    def __done_count(self) -> int:
        """
        The number of ``__next__`` calls that have been made. Exposed via ``idx``.
        """
        return 0

    @property
    def started(self):
        """
        Shortcut function to check if ``self.idx == 0``.
        """

        return self.idx != 0
