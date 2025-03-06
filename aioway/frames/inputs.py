# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import math
import typing

from aioway.blocks import Block
from aioway.schemas import TableSchema

from .frames import Frame

__all__ = ["BlockFrame"]


@dcls.dataclass(frozen=True)
class BlockFrame(Frame):
    """
    A ``Frame`` backed by a ``Block``.
    This means that it is non-distributed, and volatile.
    """

    block: Block
    """
    The underlying data of the ``Frame``.
    """

    max_batch: int = 1
    """
    The maximum batch size of the current ``Frame``.

    The ``__len__`` should be equivalent to ``max_batch``, except when index is -1.
    """ ""

    @typing.override
    def __len__(self):
        return math.ceil(len(self.block) / self.max_batch)

    @typing.override
    def __getitem__(self, idx: int) -> Block:
        start = self.max_batch * idx
        stop = start + self.max_batch
        return self.block[start:stop]

    @property
    @typing.override
    def schema(self) -> TableSchema:
        return self.block.schema

    @property
    def device(self):
        return self.block.device
