# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from abc import ABC
from collections.abc import Generator

from tensordict import TensorDict

if typing.TYPE_CHECKING:
    from ..plans import FramePlan

__all__ = ["Frame"]


@dcls.dataclass(frozen=True)
class Frame(ABC):
    """
    ``Frame`` represents a chunk / batch of heterogenious data stored in memory,
    it is one of the main physical abstractions in ``aioway`` to represent eager computation.

    Think of it as a normal ``Sequence`` of ``TensorDict``,
    where computation happens eagerly, imperatively, and the result is stored in memory.

    Each ``TensorDict`` retrieved from ``Frame`` is a minibatch of data.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Get the number of items (rows) in the current dataframe.
        """

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> TensorDict:
        """
        Get individual items from the current ``Frame``.

        Args:
            idx:
                Index to the current ``Frame``.
                Must be an integer (does not support slice input).
        """

    def __iter__(self) -> Generator[TensorDict]:
        for i in range(len(self)):
            yield self[i]

    @property
    def op(self) -> "FramePlan":
        """
        Construct an ``Plan`` that wraps around the current ``Frame``.
        The ``Plan`` calls ``iter(frame)``, producing a stream of ``TensorDict``s.
        """

        from ..plans import FramePlan

        return FramePlan(self)
