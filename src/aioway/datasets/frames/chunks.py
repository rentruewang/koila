# Copyright (c) AIoWay Authors - All Rights Reserved

"`frames.Frame`s that produce data by slicing contiguous input records."

import dataclasses as dcls
import functools
import typing

import numpy as np

from aioway import _typing
from aioway import chunks as _chunks
from aioway import meta

from . import frames

__all__ = ["ChunkFrame", "ChunkListFrame"]


@typing.final
@dcls.dataclass(frozen=True)
class ChunkFrame(frames.Frame):
    """
    A `frames.Frame` backed by a `TensorDict` (aka a batch in `aioway`).
    This means that it is non-distributed, and volatile.
    """

    data: _chunks.Chunk
    """
    The `_chunks.Chunk` source.
    """

    @typing.override
    def __len__(self) -> int:
        return len(self.data)

    @typing.override
    def _getitem(self, idx: _typing.IntArray) -> _chunks.Chunk:
        return self.data[idx]

    @property
    @typing.override
    def attrs(self) -> meta.AttrSet:
        return self.data.attrs


@typing.final
@dcls.dataclass(frozen=True)
class ChunkListFrame(frames.Frame):
    """
    A `frames.Frame` backed by a `list[_chunks.Chunk]` (aka a batch in `aioway`).
    This means that it is non-distributed, and volatile.
    """

    _chunks: list[_chunks.Chunk] = dcls.field(default_factory=list)
    """
    The `list` of `_chunks.Chunk`s.
    The data must all have the same keys and data types (#100),
    but not necessarily the same `batch_size`.
    """

    def __post_init__(self) -> None:
        is_list_of_chunks = _typing.is_list_of(_chunks.Chunk)
        if not is_list_of_chunks(self._chunks):
            raise ValueError(
                f"Expected a list of `_chunks.Chunk`s. Got {self._chunks=}"
            )

    @typing.override
    def __len__(self) -> int:
        return self._cumsum_len[-1]

    @typing.no_type_check
    def append(self, td: _chunks.Chunk, /) -> None:
        self._chunks.append(td)

    @typing.no_type_check
    def pop(self) -> _chunks.Chunk:
        return self._chunks.pop()

    @typing.override
    @typing.no_type_check
    def _getitem(self, idx: _typing.IntArray, /) -> _chunks.Chunk:
        # Which tensordict to use in `self.tensordicts`.
        td_idx: _typing.IntArray = np.searchsorted(self._cumsum_len, idx, side="right")
        assert td_idx.shape == idx.shape

        # How many elements are in the partitions prior to the current.
        prior_elements: _typing.IntArray = np.roll(self._cumsum_len, 1)
        prior_elements[0] = 0

        # Index in partition = original index - elements in prior partitions.
        idx_in_part: _typing.IntArray = idx - prior_elements[td_idx]

        # `_chunks.Chunk` that each index would correspond to.
        td_for_idx: list[_chunks.Chunk] = [self._chunks[t] for t in td_idx]

        assert len(idx_in_part) == len(td_for_idx)

        chunks: list[_chunks.Chunk] = []
        for td, part_idx in zip(td_for_idx, idx_in_part.tolist()):
            assert -len(td) <= part_idx < len(td), {
                "index for sub partition": part_idx,
                "tensordict's length": len(td),
            }
            chunks.append(td[part_idx : part_idx + 1])
        return _chunks.Chunk.cat(chunks)

    @property
    def _cumsum_len(self) -> _typing.IntArray:
        return np.cumsum([len(d) for d in self._chunks])

    @property
    @typing.override
    def attrs(self) -> meta.AttrSet:
        return self._attrs

    @functools.cached_property
    def _attrs(self):
        attrs = {chunk.attrs for chunk in self._chunks}

        if len(attrs) == 1:
            [attr] = attrs
            return attr

        raise ValueError("Chunks should have the same attrs.")
