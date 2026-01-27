# Copyright (c) AIoWay Authors - All Rights Reserved

"The binary ``Stream``s that consumes 2 ``Stream``s."

import dataclasses as dcls
import typing
from collections.abc import Generator

import tensordict
import torch
from tensordict import TensorDict
from torch import Tensor

from .sources import CacheStream
from .streams import Stream

__all__ = ["ZipStream", "NestedLoopJoinStream"]


@dcls.dataclass
class ZipStream(Stream):
    """
    ``ZipStream`` is similar to what ``zip`` does.
    """

    left: Stream
    """
    The LHS stream.
    """

    right: Stream
    """
    The RHS stream.
    """

    @property
    @typing.override
    def size(self) -> int:
        return min(self.left.size, self.right.size)

    @typing.override
    def _read(self) -> TensorDict:
        # Either one of those may raise ``StopIteration``, at which point it is done.
        left_batch = next(self.left)
        right_batch = next(self.right)

        return tensordict.merge_tensordicts(left_batch, right_batch)

    @typing.override
    def _children(self) -> Generator[Stream]:
        yield self.left
        yield self.right


@dcls.dataclass
class NestedLoopJoinStream(Stream):
    """
    This is a stream that combines 2 input streams in a nested-loop matter,
    as in ``[[x, y] for x in left for y in right if x.key == y.key]``.

    The end result would be merged with ``tensordict.merge_tensordicts``.
    """

    left: Stream
    """
    LHS is a normal stream. Will only be iterated over once.
    """

    right: CacheStream
    """
    RHS is a ``Stream`` supporting index access, thus requiring materialization.
    """

    key: str
    """
    The key to join on.
    """

    def __post_init__(self) -> None:
        # It is necessary to save the last batch for the LHS,
        # as it would be paired with multiple RHS batches.
        self.__lhs_batch: TensorDict | None = None

    @property
    @typing.override
    def size(self) -> int:
        return self.left.size * self.right.size

    @typing.override
    def _children(self) -> Generator["Stream"]:
        yield self.left
        yield self.right

    @typing.override
    def _read(self) -> TensorDict:
        lhs_batch: TensorDict = self._get_lhs()
        rhs_batch: TensorDict = self._get_rhs()

        lhs_select: Tensor = lhs_batch[self.key]
        rhs_select: Tensor = rhs_batch[self.key]

        matrix = lhs_select[:, None] == rhs_select[None, :]
        l, r = torch.nonzero(matrix).T
        assert len(l) == len(r) == torch.sum(matrix)
        out = tensordict.merge_tensordicts(lhs_batch[l], rhs_batch[r])
        assert len(out) == torch.sum(matrix)
        return out

    def _get_lhs(self) -> TensorDict:
        # Clear cache and re-evalaute.
        if self._right_idx == 0:
            self.__lhs_batch = next(self.left)
        # print("lhs outside", self.left.idx, file=open("scripts/lhs.txt", "a"))
        return self.__lhs_batch

    def _get_rhs(self) -> TensorDict:
        if self.idx < self.right.size:
            out = next(self.right)
            assert out is self.right[self._right_idx]
        else:
            assert len(self.right) == self.right.size
            assert self.left.idx > 1
        return self.right[self._right_idx]

    @property
    def _right_idx(self):
        return self.idx % self.right.size
