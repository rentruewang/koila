# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import functools
import typing
from collections.abc import Iterator

import torch
from numpy.typing import NDArray

from aioway.attrs import AttrSet
from aioway.blocks import Block
from aioway.errors import AiowayError

from .execs import Exec
from .inputs import FrameExec

if typing.TYPE_CHECKING:
    pass

__all__ = ["NestedLoopExec", "ZipExec"]


@typing.final
@dcls.dataclass(frozen=True)
class NestedLoopExec(Exec, key="NESTED_LOOP"):
    """
    The base class for ``Exec``s that are Cartesian products,
    with LHS being an unbound stream, and RHS being bounded.

    Only handles the cases where join keys are stored within the frames themselves.
    """

    left: Exec
    """
    The LHS of the operator.
    """

    right: FrameExec
    """
    The RHS of the operator.
    """

    on: str
    """
    The column for which to join.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.left, Exec):
            raise PartitionOperandTypeError(
                f"LHS should be of type exec. Got {type(self.left)=}"
            )

        if not isinstance(self.right, FrameExec):
            raise PartitionOperandTypeError(
                f"RHS should be of type frame. Got {type(self.right)=}"
            )

    @typing.override
    def __next__(self) -> Block:
        left_block, right_block = next(self._iterator)
        left_select, right_select = self._join(left=left_block, right=right_block)

        left_chosen = left_block[left_select]
        right_chosen = right_block[right_select]

        return left_chosen.zip(right_chosen)

    @property
    @typing.override
    def attrs(self) -> AttrSet:
        return self.left.attrs | self.right.attrs

    @property
    @typing.override
    def children(self) -> tuple[Exec, FrameExec]:
        return self.left, self.right

    @functools.cached_property
    def _iterator(self) -> Iterator[tuple[Block, Block]]:
        """
        The actual iterator that will be used to iterate over the LHS.
        """

        return self._nested_loop()

    def _nested_loop(self):
        for left_block in self.left:
            self.right.reset()
            for right_block in self.right:
                yield left_block, right_block

    def _join(self, left: Block, right: Block) -> tuple[NDArray, NDArray]:
        return self._compute_matching(left=left, right=right, on=self.on)

    @staticmethod
    def _compute_matching(
        left: Block, right: Block, on: str
    ) -> tuple[NDArray, NDArray]:
        left_key = left[on]
        right_key = right[on]

        matrix = left_key[:, None] == right_key[None, :]
        l, r = torch.nonzero(matrix).cpu().numpy().T
        return l, r


@typing.final
@dcls.dataclass(frozen=True)
class ZipExec(Exec, key="ZIP"):
    """
    ``ZipExec`` merges 2 ``Exec``s that have identical length together.
    """

    left: Exec
    right: Exec

    def __post_init__(self) -> None:
        # Check intersection with the logic in ``TableSchema.__and__``.
        _ = self.left.attrs & self.right.attrs

    @typing.override
    def __next__(self) -> Block:
        left = next(self.left)
        right = next(self.right)
        return left.zip(right)

    @property
    @typing.override
    def attrs(self) -> AttrSet:
        return self.left.attrs | self.right.attrs

    @property
    @typing.override
    def children(self) -> tuple[Exec, Exec]:
        return self.left, self.right


class PartitionOperandTypeError(AiowayError, TypeError): ...


class ConcatLengthMismatchError(AiowayError, TypeError): ...
