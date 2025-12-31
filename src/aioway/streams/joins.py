# Copyright (c) AIoWay Authors - All Rights Reserved

"The binary ``Stream``s that consumes 2 ``Stream``s."

import dataclasses as dcls
import typing
from collections.abc import Generator

import tensordict
import torch
from tensordict import TensorDict
from torch import Tensor

from .streams import Stream

if typing.TYPE_CHECKING:
    from aioway.tables import TableStream

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

    @typing.override
    def __len__(self) -> int:
        return min(len(self.left), len(self.right))

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

    right: "TableStream"
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

    @typing.override
    def __len__(self) -> int:
        return len(self.left) * len(self.right)

    @typing.override
    def _children(self) -> Generator["Stream"]:
        yield self.left
        yield self.right

    @typing.override
    def _read(self) -> TensorDict:
        if self._rhs_done_or_not_started():
            # Reset the iteration on the RHS.
            self.right = self.right.reset()

            # Clear cache and re-evalaute.
            self.__lhs_batch = None

        rhs_batch = self.right[self.idx % len(self.right)]
        lhs_batch = self._get_lhs()

        lhs_select: Tensor = lhs_batch[self.key]
        rhs_select: Tensor = rhs_batch[self.key]

        matrix = lhs_select[:, None] == rhs_select[None, :]
        l, r = torch.nonzero(matrix).T
        return tensordict.merge_tensordicts(lhs_batch[l], rhs_batch[r])

    def _rhs_done_or_not_started(self) -> bool:
        return self.idx % len(self.right) == 0

    def _get_lhs(self) -> TensorDict:
        if self.__lhs_batch is None:
            self.__lhs_batch = next(self.left)
        return self.__lhs_batch
