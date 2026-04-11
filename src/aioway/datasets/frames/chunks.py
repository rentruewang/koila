# Copyright (c) AIoWay Authors - All Rights Reserved

"`Frame`s that produce data by slicing contiguous input records."

import dataclasses as dcls
import functools
import typing

import numpy as np

from aioway._common import IntArray, is_list_of
from aioway.chunks import Chunk
from aioway.schemas import AttrSet

from .frames import Frame

__all__ = ["ChunkFrame", "ChunkListFrame"]


@typing.final
@dcls.dataclass(frozen=True)
class ChunkFrame(Frame):
    """
    A `Frame` backed by a `TensorDict` (aka a batch in `aioway`).
    This means that it is non-distributed, and volatile.
    """

    data: Chunk
    """
    The `Chunk` source.
    """

    @typing.override
    def __len__(self) -> int:
        return len(self.data)

    @typing.override
    def _getitem(self, idx: IntArray) -> Chunk:
        return self.data[idx]

    @property
    @typing.override
    def attrs(self) -> AttrSet:
        return self.data.attrs


@typing.final
@dcls.dataclass(frozen=True)
class ChunkListFrame(Frame):
    """
    A `Frame` backed by a `list[Chunk]` (aka a batch in `aioway`).
    This means that it is non-distributed, and volatile.
    """

    _chunks_list: list[Chunk] = dcls.field(default_factory=list)
    """
    The `list` of `Chunk`s.
    The data must all have the same keys and data types (#100),
    but not necessarily the same `batch_size`.
    """

    def __post_init__(self) -> None:
        is_list_of_chunks = is_list_of(Chunk)
        if not is_list_of_chunks(self._chunks_list):
            raise ValueError(f"Expected a list of `Chunk`s. Got {self._chunks_list=}")

    @typing.override
    def __len__(self) -> int:
        return self._cumsum_len[-1]

    def append(self, td: Chunk, /) -> None:
        self._chunks_list.append(td)

    def pop(self) -> Chunk:
        return self._chunks_list.pop()

    @typing.override
    @typing.no_type_check
    def _getitem(self, idx: IntArray, /) -> Chunk:
        # Which tensordict to use in `self.tensordicts`.
        td_idx: IntArray = np.searchsorted(self._cumsum_len, idx, side="right")
        assert td_idx.shape == idx.shape

        # How many elements are in the partitions prior to the current.
        prior_elements: IntArray = np.roll(self._cumsum_len, 1)
        prior_elements[0] = 0

        # Index in partition = original index - elements in prior partitions.
        idx_in_part: IntArray = idx - prior_elements[td_idx]

        # `Chunk` that each index would correspond to.
        td_for_idx: list[Chunk] = [self._chunks_list[t] for t in td_idx]

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
        return np.cumsum([len(d) for d in self._chunks_list])

    @property
    @typing.override
    def attrs(self) -> AttrSet:
        return self._attrs

    @functools.cached_property
    def _attrs(self):
        attrs = {chunk.attrs for chunk in self._chunks_list}

        if len(attrs) == 1:
            [attr] = attrs
            return attr

        raise ValueError("Chunks should have the same attrs.")
