# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import functools
import typing
from abc import ABC

import tensordict
from tensordict import TensorDict
from torch.utils.data import Dataset

from aioway.attrs import AttrSet

if typing.TYPE_CHECKING:
    from aioway.frames.indices import IndexManager

__all__ = ["Frame"]


@dcls.dataclass(frozen=True)
class Frame(Dataset[TensorDict], ABC):
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

        ...

    @abc.abstractmethod
    def __getitem__(self, idx: int, /) -> TensorDict:
        """
        Get individual items from the current ``Frame``.
        """

        ...

    @abc.abstractmethod
    def __getitems__(self, idx: list[int], /) -> TensorDict:
        """
        Get a batch of items at once from the current ``Frame``.
        """

        return tensordict.stack([self[i] for i in idx], dim=0)

    def __str__(self) -> str:
        return repr(self)

    @property
    @abc.abstractmethod
    def attrs(self) -> AttrSet: ...

    @functools.cache
    def index(self) -> "IndexManager":
        from aioway.frames.indices import IndexManager

        return IndexManager(self)
