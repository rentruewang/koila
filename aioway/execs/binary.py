# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import math
import typing
from itertools import count as Count

import torch
from numpy.typing import NDArray

from aioway.attrs import AttrSet
from aioway.blocks import Block
from aioway.errors import AiowayError
from aioway.frames.frames import Frame

from .execs import Exec

if typing.TYPE_CHECKING:
    from aioway.frames import Frame

__all__ = ["MatrixJoinExec", "ZipExec"]


@typing.final
@dcls.dataclass
class MatrixJoinExec(Exec, key="MATRIX_JOIN"):
    """
    The base class for ``Exec``s that are Cartesian products,
    with LHS being an unbound stream, and RHS being bounded.

    Only handles the cases where join keys are stored within the frames themselves.
    """

    left: Exec
    """
    The LHS of the operator.
    """

    right: "Frame"
    """
    The RHS of the operator.
    """

    on: str
    """
    The column for which to join.
    """

    rhs_batch: int
    """
    The batch size for RHS.
    """

    __last_left_block: Block = dcls.field(init=False)
    """
    The last left block.
    """

    __counter: Count = dcls.field(init=False, default_factory=Count)
    """
    The number of iterations so far for this stream.
    """

    def __post_init__(self) -> None:
        # Import here to prevent circular depedency.
        from aioway.frames import Frame

        if not isinstance(self.left, Exec):
            raise PartitionOperandTypeError(
                f"LHS should be of type exec. Got {type(self.left)=}"
            )

        if not isinstance(self.right, Frame):
            raise PartitionOperandTypeError(
                f"RHS should be of type frame. Got {type(self.right)=}"
            )

    def __next__(self) -> Block:
        # Looped over right in the last iteration.
        loop_idx = next(self.__counter)

        num_rhs_batches = math.ceil(len(self.right) / self.rhs_batch)

        rhs_batch_idx = loop_idx % num_rhs_batches

        rhs_start = rhs_batch_idx * self.rhs_batch
        rhs_end = rhs_start + self.rhs_batch

        if rhs_batch_idx == 0:
            self.__last_left_block = next(self.left)

        rhs_keys = list(range(len(self.right))[rhs_start:rhs_end])
        right_batches = self.right.__getitems__(rhs_keys)

        left_select, right_select = self._join(self.__last_left_block, right_batches)

        left_chosen = self.__last_left_block[left_select]
        right_chosen = right_batches[right_select]

        return left_chosen.zip(right_chosen)

    @property
    @typing.override
    def attrs(self) -> AttrSet:
        return self.left.attrs | self.right.attrs

    @property
    @typing.override
    def children(self) -> tuple[Exec, Frame]:
        return self.left, self.right

    def _join(self, left: Block, right: Block) -> tuple[NDArray, NDArray]:
        return self.compute_matching(left=left, right=right, on=self.on)

    @staticmethod
    def compute_matching(left: Block, right: Block, on: str) -> tuple[NDArray, NDArray]:
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
