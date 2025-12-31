# Copyright (c) AIoWay Authors - All Rights Reserved

"The ``Table`` interface."

import abc
import dataclasses as dcls
import typing
from abc import ABC
from collections.abc import Generator

from tensordict import TensorDict

__all__ = ["Table"]


@dcls.dataclass(frozen=True)
class Table(ABC):
    """
    ``Table`` represents a chunk / batch of heterogenious data stored in memory,
    it is one of the main physical abstractions in ``aioway`` to represent eager computation.

    Think of it as a normal ``Sequence`` of ``TensorDict``,
    where computation happens eagerly, imperatively, and the result is stored in memory.

    Each ``TensorDict`` retrieved from ``Table`` is a minibatch of data.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Get the number of items (rows) in the current dataframe.
        """

    @typing.final
    def __getitem__(self, idx: int) -> TensorDict:
        """
        Get individual items from the current ``Frame``.

        Args:
            idx:
                Index to the current ``Frame``.
                Must be an integer (does not support slice input).
                Should be in the range ``[-len, len)``.

        Returns:
            A ``TensorDict`` representing a batch of data.
        """

        length = len(self)

        if not -length <= idx < length:
            raise IndexError(
                f"Index must be in the range `[-{length}, {length})`, but got {idx=}"
            )

        return self._getitem(idx % length)

    def __bool__(self) -> bool:
        return bool(len(self))

    @abc.abstractmethod
    def _getitem(self, idx: int, /) -> TensorDict:
        """
        The implementation of ``__getitem__``.

        Args:
            idx: The index being passed in. Positive, in ``[0, len)``.

        Returns:
            A batch of data.
        """

        ...

    def __iter__(self) -> Generator[TensorDict]:
        for i in range(len(self)):
            yield self[i]

    def stream(self):
        """
        Convert the current ``Table`` into a ``Stream`` for iteration.

        Returns:
            A ``Stream`` for iteration purposes.
        """

        from aioway.tables import TableStream

        return TableStream(table=self)
