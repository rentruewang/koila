# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing

import torch
from numpy.typing import NDArray

from aioway.blocks import Block

from .ops import Op2

__all__ = ["ZipOp", "MatchOp"]


@dcls.dataclass(frozen=True)
class ZipOp(Op2, key="ZIP_2"):
    """
    The ``ZIP`` operation, similar to how you use the builtin ``zip``.
    """

    @typing.override
    def join(self, left: Block, right: Block) -> Block:
        return left.zip(right)


@dcls.dataclass(frozen=True)
class MatchOp(Op2, key="MATCH_2"):
    """
    The ``MATCH`` operation yields the block with matching id.
    """

    on_left: str
    """
    The column used for index for the left block.
    """

    on_right: str
    """
    The column used for index for the right block.
    """

    @typing.override
    def join(self, left: Block, right: Block) -> Block:
        left_select, right_select = self._compute_join(left=left, right=right)

        left_chosen = left[left_select]
        right_chosen = right[right_select]

        return left_chosen.zip(right_chosen)

    def _compute_join(self, left: Block, right: Block) -> tuple[NDArray, NDArray]:
        left_key = left[self.on_left]
        right_key = right[self.on_right]

        matrix = left_key[:, None] == right_key[None, :]
        l, r = torch.nonzero(matrix).cpu().numpy().T
        return l, r
