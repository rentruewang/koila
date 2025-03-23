# Copyright (c) RenChu Wang - All Rights Reserved

import abc
from abc import ABC

from tensordict import TensorDict
from torch.utils.data import IterableDataset

from aioway.datatypes import AttrSet
from aioway.execs import IteratorExec

__all__ = ["Stream"]


class Stream(IterableDataset[TensorDict], ABC):
    @abc.abstractmethod
    def __iter__(self) -> TensorDict: ...

    @property
    @abc.abstractmethod
    def attrs(self) -> AttrSet: ...

    def iterate(self, **kwargs) -> IteratorExec:
        from aioway.tabular import iters

        iterator = iters.tabular_iterator(self, **kwargs)
        return IteratorExec(iterator, self.attrs)
