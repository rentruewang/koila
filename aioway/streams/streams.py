# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import typing
from abc import ABC

from tensordict import TensorDict
from torch.utils.data import IterableDataset

from aioway.attrs import AttrSet
from aioway.plans import PhysicalPlan

__all__ = ["Stream"]


class Stream(IterableDataset[TensorDict], PhysicalPlan, ABC):
    @abc.abstractmethod
    def __iter__(self) -> TensorDict: ...

    def __str__(self) -> str:
        return repr(self)

    @property
    @abc.abstractmethod
    def attrs(self) -> AttrSet: ...

    @property
    @typing.override
    def children(self) -> tuple[()]:
        """
        ``Frame``s and ``Stream``s are by definition, sources of the data flow.
        Therefore, they have no children.
        """

        return ()
