# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing

import torch
from numpy.typing import NDArray

from aioway.blocks import Block

from .ops import BlockIter, Op2

__all__ = ["ZipOp", "MatchOp"]


@dcls.dataclass(frozen=True)
class ZipOp(Op2, key="ZIP"):
    """
    The ``ZIP`` operation, similar to how you use the builtin ``zip``.
    """

    zip = zip
    "Same as built in ``zip``."

    join = lambda self, left, right: left.zip(right)
    "Joins the left and right that are the same length."


@dcls.dataclass(frozen=True)
class MatchOp(Op2, key="MATCH"):
    """
    The ``MATCH`` operation yields the block with matching id.
    """

    key: str
    """
    The column used for index for the left and right block.
    """

    @typing.override
    def zip(self, left_iter: BlockIter, right_iter: BlockIter):
        "Same as ``iteratools.product``, but repeatedly invoke ``iter(right)``."

        for left in left_iter:
            for right in right_iter:
                yield left, right

    @typing.override
    def join(self, left: Block, right: Block) -> Block:
        "Inner join a block of data in memory."

        left_select, right_select = self._compute_select(left=left, right=right)

        left_chosen = left[left_select]
        right_chosen = right[right_select]

        return left_chosen.zip(right_chosen)

    def _compute_select(self, left: Block, right: Block) -> tuple[NDArray, NDArray]:
        left_key = left[self.key]
        right_key = right[self.key]

        matrix = left_key[:, None] == right_key[None, :]
        l, r = torch.nonzero(matrix).cpu().numpy().T
        return l, r
