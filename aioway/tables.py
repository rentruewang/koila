# Copyright (c) RenChu Wang - All Rights Reserved

import abc
from abc import ABC
from collections.abc import Iterable

from aioway.datatypes import AttrSet
from aioway.blocks import Block

__all__ = ["Table"]


class Table(Iterable[Block], ABC):
    @property
    @abc.abstractmethod
    def attrs(self) -> AttrSet:
        """
        The output schema of the current table.
        """

        ...

    @property
    def inputs(self) -> tuple["Table", ...]:
        raise NotImplementedError("This needs to be done.")
