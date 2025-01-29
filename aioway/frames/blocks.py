# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls

import numpy as np

from aioway.blocks import Block
from aioway.buffers import Buffer

from .frames import Frame

__all__ = ["BlockFrame"]


@dcls.dataclass(frozen=True)
class BlockFrame(Frame):
    """
    ``BlockFrame`` is a ``Frame`` backed by a ``Block``.
    This means that a ``BlockFrame`` is entirely in-memory,
    and support type conversions between different ``Block`` backends.

    Fixme:
        For ``cols``, use ``Buffer`` instead,
        do not use this hack of converting to numpy.
    """

    block: Block
    """
    Underlying data of ``BlockFrame``.
    """

    def count(self) -> int:
        return len(self.block)

    def cols(self, key: str) -> Buffer:
        return self.block[key]

    def _rows(self, idx: list[int]) -> Block:
        return self.block[np.array(idx)]
