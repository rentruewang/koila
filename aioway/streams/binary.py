# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
import operator
import typing
from itertools import count as Count

import torch
from numpy.typing import NDArray

from aioway.blocks import Block
from aioway.errors import AiowayError
from aioway.schemas import TableSchema
from aioway.streams.streams import Stream

if typing.TYPE_CHECKING:
    from aioway.frames.frames import Frame

__all__ = ["MatrixJoinStream", "ZipStream"]


@dcls.dataclass
class _PartialCartesianStream(Stream):
    """
    The base class for ``Stream``s that are Cartesian products,
    with LHS being an unbound stream, and RHS being bounded.

    Only handles the cases where join keys are stored within the frames themselves.
    """

    left: Stream
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

    @typing.override
    def __next__(self) -> Block:
        # Looped over right in the last iteration.
        if (right_idx := next(self.__counter) % len(self.right)) == 0:
            self.__last_left_block = next(self.left)

        right_item = self.right[right_idx]

        left_select, right_select = self._join(self.__last_left_block, right_item)
        return self.__last_left_block[left_select].zip(right_item[right_select])

    @property
    def schema(self) -> TableSchema:
        return self.left.schema | self.right.schema

    @abc.abstractmethod
    def _join(self, left: Block, right: Block) -> tuple[NDArray, NDArray]: ...


@typing.final
@dcls.dataclass
class MatrixJoinStream(_PartialCartesianStream):
    @typing.override
    def __length_hint__(self):
        return operator.length_hint(self.left) * len(self.right)

    def _join(self, left: Block, right: Block) -> tuple[NDArray, NDArray]:
        left_key = left[self.on]
        right_key = right[self.on]

        matrix = left_key[:, None] == right_key[None, :]
        l, r = torch.nonzero(matrix).cpu().numpy().T
        return l, r


@typing.final
@dcls.dataclass(frozen=True)
class ZipStream(Stream):
    """
    ``ConcatStream`` merges 2 ``Stream``s that have identical length together.
    """

    left: Stream
    right: Stream

    def __post_init__(self) -> None:
        # Check intersection with the logic in ``TableSchema.__and__``.
        _ = self.left.schema & self.right.schema

    @typing.override
    def __length_hint__(self):
        left_hint = operator.length_hint(self.left)
        right_hint = operator.length_hint(self.right)

        # If ``__legnth_hint__`` is not defined,
        # ``operator.length_hint`` would be 0.

        if left_hint and right_hint:
            return min(left_hint, right_hint)

        if not left_hint and not right_hint:
            return NotImplemented

        return left_hint or right_hint

    @typing.override
    def __next__(self) -> Block:
        left = next(self.left)
        right = next(self.right)
        return left.zip(right)

    @property
    def schema(self) -> TableSchema:
        return self.left.schema | self.right.schema


class ConcatLengthMismatchError(AiowayError, TypeError): ...
