# Copyright (c) RenChu Wang - All Rights Reserved

import abc
from abc import ABC
from collections.abc import Iterable, Iterator

from numpy.typing import NDArray
from tensordict import TensorDict

__all__ = ["Frame"]


class Frame(ABC):
    @abc.abstractmethod
    def count(self) -> int:
        """
        Get the number of items (rows) in the current dataframe.
        """

        ...

    @abc.abstractmethod
    def cols(self, key: str) -> NDArray:
        """
        Get the selected column in numpy array format.
        """

        ...

    @abc.abstractmethod
    def rows(self, idx: Iterable[int]) -> TensorDict:
        """
        Random access for the indices.

        Args:
            idx: An ``Iterable`` of indices to get the rows from.

        Returns:
            A tensordict representing the data of the selected batch.

        Todo:
            Swap out the tensordict to a ``Block``.
        """

        ...

    def iterator(self, batch: int) -> Iterator[TensorDict]:
        for idx in range(self.count()):
            idx_range = range(idx, idx + batch)
            yield self.rows(idx_range)
