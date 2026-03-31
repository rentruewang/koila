# Copyright (c) AIoWay Authors - All Rights Reserved

"`StreamColumn`s are a column of `streams.Stream`."

import dataclasses as dcls
import typing
from collections import abc as cabc

from aioway import chunks

from .. import datasets
from . import streams

__all__ = ["StreamColumnView", "StreamSelectView"]


@dcls.dataclass(frozen=True)
class StreamColumnView(
    cabc.Iterator[chunks.Vector], datasets.DatasetColumnView[streams.Stream]
):
    """
    A column reference (on a stream).
    Performs `__next__` and yield `chunks.Vector`s.
    """

    def __iter__(self) -> typing.Self:
        return self

    def __next__(self) -> chunks.Vector:
        batch = next(self.dset)
        return batch[self.col]

    @classmethod
    def from_column(cls, dataset: streams.Stream, /, column: str) -> typing.Self:
        return cls(col=column, dset=dataset)


@dcls.dataclass(frozen=True)
class StreamSelectView(datasets.DatasetSelectView[streams.Stream], streams.Stream):
    """
    The view generated when calling `streams.Stream.select`.
    """

    COLUMN_TYPE = StreamColumnView

    def __iter__(self) -> typing.Self:
        return self

    @typing.override
    def _next(self) -> chunks.Chunk:
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
    def from_columns(cls, dataset: streams.Stream, /, *columns: str) -> typing.Self:
        return cls(dset=dataset, cols=columns)
