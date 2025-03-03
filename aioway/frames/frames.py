# Copyright (c) RenChu Wang - All Rights Reserved

import abc
from abc import ABC
from collections.abc import Iterable, Sequence

from tensordict import TensorDict

from aioway.errors import AiowayError

__all__ = ["Frame"]


class Frame(Sequence[TensorDict], ABC):
    """
    ``Frame`` represents a chunk / batch of heterogenious data stored in memory,
    it is one of the main physical abstractions in ``aioway`` to represent eager computation.

    Think of it as a normal ``Sequence`` of ``TensorDict``,
    where computation happens eagerly, imperatively, and the result is stored in memory.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Get the number of items (rows) in the current dataframe.
        """

        ...

    def __getitem__(self, idx: int | slice | Iterable[int]) -> TensorDict:
        """
        Get individual items from the current dataframe.
        """

        if isinstance(idx, int):
            return self._rows_int(idx)

        if isinstance(idx, slice):
            return self._rows_slice(idx)

        if isinstance(idx, Iterable):
            return self._rows_list(list(idx))

        raise FrameGetItemTypeError(
            f"Unknown type: {type(idx)=}. Must be int or slice or iterable"
        )

    @abc.abstractmethod
    def _rows_int(self, idx: int, /) -> TensorDict: ...

    @abc.abstractmethod
    def _rows_slice(self, idx: slice, /) -> TensorDict: ...

    @abc.abstractmethod
    def _rows_list(self, idx: list[int], /) -> TensorDict: ...


class FrameGetItemTypeError(AiowayError, TypeError): ...
