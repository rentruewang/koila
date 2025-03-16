# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from abc import ABC

from tensordict import TensorDict
from torch.utils.data import Dataset

from aioway.blocks import Block
from aioway.streams import IteratorStream, Stream
from aioway.tables import Table

__all__ = ["Frame"]


@dcls.dataclass(frozen=True)
class Frame(Dataset[TensorDict], Table, ABC):
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
    def __getitem__(self, idx: int) -> Block:
        """
        Get individual items from the current dataframe.
        """

        ...

    @typing.override
    def __iter__(self) -> Stream:
        def generator():
            for idx in range(len(self)):
                yield self[idx]

        return IteratorStream(iter(generator()), self.attrs)
