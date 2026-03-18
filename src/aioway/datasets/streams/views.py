# Copyright (c) AIoWay Authors - All Rights Reserved

"`StreamColumn`s are a column of `Stream`."

import dataclasses as dcls
import typing
from collections.abc import Iterator
from typing import Self

from aioway.chunks import Chunk, Vector

from ..datasets import DatasetColumnView, DatasetSelectView
from .streams import Stream

__all__ = ["StreamColumnView", "StreamSelectView"]


@dcls.dataclass(frozen=True)
class StreamColumnView(Iterator[Vector], DatasetColumnView[Stream]):
    """
    A column reference (on a stream).
    Performs `__next__` and yield `Vector`s.
    """

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Vector:
        batch = next(self.dset)
        return batch[self.col]

    @classmethod
    def from_column(cls, dataset: Stream, /, column: str) -> Self:
        return cls(col=column, dset=dataset)


@dcls.dataclass(frozen=True)
class StreamSelectView(DatasetSelectView[Stream], Stream):
    """
    The view generated when calling `Stream.select`.
    """

    COLUMN_TYPE = StreamColumnView

    def __iter__(self) -> Self:
        return self

    @typing.override
    def _compute(self) -> Chunk:
        batch = next(self.dset)
        return batch.select(*self.cols)

    @property
    @typing.override
    def size(self) -> int:
        return self.dset.size

    @typing.override
    def _inputs(self):
        return (self.dset,)

    @classmethod
    def from_columns(cls, dataset: Stream, /, *columns: str) -> Self:
        return cls(dset=dataset, cols=columns)
