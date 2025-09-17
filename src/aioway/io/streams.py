# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
from abc import ABC

from tensordict import TensorDict
from torch.utils.data import IterableDataset

from aioway.attrs import AttrSet

__all__ = ["Stream"]


class Stream(IterableDataset[TensorDict], ABC):
    @abc.abstractmethod
    def __iter__(self) -> TensorDict: ...

    def __str__(self) -> str:
        return repr(self)

    @property
    @abc.abstractmethod
    def attrs(self) -> AttrSet: ...
