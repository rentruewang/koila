# Copyright (c) AIoWay Authors - All Rights Reserved

"The binary `Stream`s that consumes 2 `Stream`s."

import dataclasses as dcls
import typing

import torch

from aioway.chunks import Chunk
from aioway.tdicts import AttrSet

from .sources import CacheStream
from .streams import Stream, StreamState

__all__ = ["ZipStream", "NestedLoopJoinStream"]


@dcls.dataclass(frozen=True)
class ZipStream(Stream):
    """
    `ZipStream` is similar to what `zip` does.
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

    @property
    @typing.override
    def attrs(self) -> AttrSet:
        return self.left.attrs | self.right.attrs

    @typing.override
    def _next(self) -> Chunk:
        # Either one of those may raise `StopIteration`, at which point it is done.
        left_batch = next(self.left)
        right_batch = next(self.right)

        return left_batch.zip(right_batch)

    @typing.override
    def _inputs(self):
        return self.left, self.right


@dcls.dataclass
class NestedState(StreamState):
    lhs_batch: Chunk | None = None

    # It is necessary to save the last batch for the LHS,
    # as it would be paired with multiple RHS batches.


@dcls.dataclass(frozen=True)
class NestedLoopJoinStream(Stream):
    """
    This is a stream that combines 2 input streams in a nested-loop matter,
    as in `[[x, y] for x in left for y in right if x.key == y.key]`.

    The end result would be merged with `tensordict.merge_tensordicts`.
    """

    left: Stream
    """
    LHS is a normal stream. Will only be iterated over once.
    """

    right: CacheStream
    """
    RHS is a `Stream` supporting index access, thus requiring materialization.
    """

    key: str
    """
    The key to join on.
    """

    state: NestedState = dcls.field(default_factory=NestedState)

    @property
    @typing.override
    def size(self) -> int:
        return self.left.size * self.right.size

    @property
    @typing.override
    def attrs(self) -> AttrSet:
        return self.left.attrs | self.right.attrs

    @typing.override
    def _inputs(self):
        return self.left, self.right

    @typing.override
    def _next(self) -> Chunk:
        lhs_batch = self._get_lhs()
        rhs_batch = self._get_rhs()

        lhs_select = lhs_batch[self.key]
        rhs_select = rhs_batch[self.key]

        matrix = lhs_select.torch()[:, None] == rhs_select.torch()[None, :]
        l, r = torch.nonzero(matrix).T
        assert len(l) == len(r) == torch.sum(matrix)
        out = lhs_batch[l].zip(rhs_batch[r])
        assert len(out) == torch.sum(matrix)
        return out

    def _get_lhs(self) -> Chunk:
        # Clear cache and re-evalaute.
        if self._right_idx == 0:
            self.state.lhs_batch = next(self.left)
        assert self.state.lhs_batch
        return self.state.lhs_batch

    def _get_rhs(self) -> Chunk:
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
