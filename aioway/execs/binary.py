# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from itertools import count as Count

import torch
from numpy.typing import NDArray

from aioway.blocks import Block
from aioway.datatypes import AttrSet
from aioway.errors import AiowayError

from .execs import Exec

if typing.TYPE_CHECKING:
    from aioway.frames import Frame

__all__ = ["MatrixJoinExec", "ZipExec"]


@dcls.dataclass
class _PartialCartesianExec(Exec):
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

    @typing.override
    def __next__(self) -> Block:
        # Looped over right in the last iteration.
        if (right_idx := next(self.__counter) % len(self.right)) == 0:
            self.__last_left_block = next(self.left)

        right_item = self.right[right_idx]

        left_select, right_select = self._join(self.__last_left_block, right_item)
        return self.__last_left_block[left_select].zip(right_item[right_select])

    @property
    def attrs(self) -> AttrSet:
        return self.left.attrs | self.right.attrs

    @abc.abstractmethod
    def _join(self, left: Block, right: Block) -> tuple[NDArray, NDArray]: ...


@typing.final
@dcls.dataclass
class MatrixJoinExec(_PartialCartesianExec):
    def _join(self, left: Block, right: Block) -> tuple[NDArray, NDArray]:
        left_key = left[self.on]
        right_key = right[self.on]

        matrix = left_key[:, None] == right_key[None, :]
        l, r = torch.nonzero(matrix).cpu().numpy().T
        return l, r


@typing.final
@dcls.dataclass(frozen=True)
class ZipExec(Exec):
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
    def attrs(self) -> AttrSet:
        return self.left.attrs | self.right.attrs


class PartitionOperandTypeError(AiowayError, TypeError): ...


class ConcatLengthMismatchError(AiowayError, TypeError): ...
