# Copyright (c) RenChu Wang - All Rights Reserved

import abc
from abc import ABC
from collections.abc import Iterable

from aioway.blocks import Block
from aioway.buffers import Buffer

__all__ = ["Frame"]


class Frame(ABC):
    """
    ``Frame`` represents a chunk / batch of heterogenious data stored in memory,
    it is one of the main physical abstractions in ``aioway`` to represent eager computation.

    Think of it as a normal ``pandas.DataFrame`` or ``torch.Tensor`` or ``TensorDict``,
    where computation happens eagerly, imperatively, and the result is stored in memory.

    Todo:
        I have decided that ``Frame`` is abstract,
        and that it represents bounded, in-memory dataframe.

        This means that memory layouts like ``arrow``, ``pandas`` can easily be supported.

        However, to be fast, instead of serializing to python objects,
        we serialize to a concrete ``Batch`` object (to be introduced),
        that is a thin wrapper over ``TensorDict``, representing the current batch,
        allowing computing on GPUs.

        This way, UDFs can still be implemented, but native methods can be used as well.
    """

    @abc.abstractmethod
    def count(self) -> int:
        """
        Get the number of items (rows) in the current dataframe.
        """

        ...

    @abc.abstractmethod
    def cols(self, key: str) -> Buffer:
        """
        Get the selected column in numpy array format.
        """

        ...

    def rows(self, idx: Iterable[int]) -> Block:
        """
        Random access for the indices.

        Args:
            idx: An ``Iterable`` of indices to get the rows from.

        Returns:
            A tensordict representing the data of the selected batch.
        """

        return self._rows(list(idx))

    @abc.abstractmethod
    def _rows(self, idx: list[int]) -> Block: ...
