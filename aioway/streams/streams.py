# Copyright (c) RenChu Wang - All Rights Reserved

import abc
from abc import ABC

from tensordict import TensorDict
from torch.utils.data import IterableDataset

from aioway.datatypes import AttrSet

__all__ = ["Stream"]


class Stream(IterableDataset[TensorDict], ABC):
    @abc.abstractmethod
    def __iter__(self) -> TensorDict: ...

    @property
    @abc.abstractmethod
    def attrs(self) -> AttrSet: ...
