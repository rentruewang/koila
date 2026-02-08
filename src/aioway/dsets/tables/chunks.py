# Copyright (c) AIoWay Authors - All Rights Reserved

"``Table``s that produce data by slicing contiguous input records."

import dataclasses as dcls
import typing
from typing import TypeIs

import numpy as np

from aioway.chunks import Chunk

from .tables import IntArray, Table

__all__ = ["ChunkTable", "ChunkListTable"]


@dcls.dataclass(frozen=True)
class ChunkTable(Table):
    """
    A ``Table`` backed by a ``TensorDict`` (aka a batch in ``aioway``).
    This means that it is non-distributed, and volatile.
    """

    data: Chunk
    """
    The ``Chunk`` source.
    """

    @typing.override
    def __len__(self) -> int:
        return len(self.data)

    @typing.override
    def _getitem(self, idx: IntArray) -> Chunk:
        return self.data[idx]


@typing.final
@dcls.dataclass(frozen=True)
class ChunkListTable(Table):
    """
    A ``Table`` backed by a ``list[Chunk]`` (aka a batch in ``aioway``).
    This means that it is non-distributed, and volatile.
    """

    chunks: list[Chunk] = dcls.field(default_factory=list)
    """
    The ``list`` of ``Chunk``s.
    The data must all have the same keys and data types (#100),
    but not necessarily the same ``batch_size``.
    """

    def __post_init__(self) -> None:
        if not is_list_of_chunks(self.chunks):
            raise ValueError(f"Expected a list of `Chunk`s. Got {self.chunks=}")

    @typing.override
    def __len__(self) -> int:
        return self._cumsum_len[-1]

    def append(self, td: Chunk, /) -> None:
        self.chunks.append(td)

    def pop(self) -> Chunk:
        return self.chunks.pop()

    @typing.override
    def _getitem(self, idx: IntArray, /) -> Chunk:
        # Which tensordict to use in ``self.tensordicts``.
        td_idx: IntArray = np.searchsorted(self._cumsum_len, idx, side="right")
        assert td_idx.shape == idx.shape

        # How many elements are in the partitions prior to the current.
        prior_elements: IntArray = np.roll(self._cumsum_len, 1)
        prior_elements[0] = 0

        # Index in partition = original index - elements in prior partitions.
        idx_in_part: IntArray = idx - prior_elements[td_idx]

        # ``Chunk`` that each index would correspond to.
        td_for_idx: list[Chunk] = [self.chunks[t] for t in td_idx]

        assert len(idx_in_part) == len(td_for_idx)

        chunks: list[Chunk] = []
        for td, part_idx in zip(td_for_idx, idx_in_part.tolist()):
            assert -len(td) <= part_idx < len(td), {
                "index for sub partition": part_idx,
                "tensordict's length": len(td),
            }
            chunks.append(td[part_idx : part_idx + 1])
        return Chunk.cat(chunks)

    @property
    def _cumsum_len(self) -> IntArray:
        return np.cumsum([len(d) for d in self.chunks])


def is_list_of_chunks(data) -> TypeIs[list[Chunk]]:
    return isinstance(data, list) and all(isinstance(t, Chunk) for t in data)
