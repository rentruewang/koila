# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import inspect
import typing
from abc import ABC
from collections.abc import Generator

from tensordict import TensorDict

from aioway import _registries

if typing.TYPE_CHECKING:
    from aioway.ops import FrameOp

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

    def __init_subclass__(cls, key: str = ""):
        if inspect.isabstract(cls):
            return

        init_sublcass = _registries.init_subclass(lambda: Frame)
        init_sublcass(cls, key=key)

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
    def op(self) -> "FrameOp":
        """
        Construct an ``Op`` that wraps around the current ``Frame``.
        The ``Op`` calls ``iter(frame)``, producing a stream of ``TensorDict``s.
        """

        from aioway.ops import FrameOp

        return FrameOp(self)
