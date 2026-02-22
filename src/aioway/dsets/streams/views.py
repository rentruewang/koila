# Copyright (c) AIoWay Authors - All Rights Reserved

"``StreamColumn``s are a column of ``Stream``."

import dataclasses as dcls
import typing
from collections.abc import Generator, Iterator, Sequence
from typing import Self

from aioway.attrs import Attr, AttrSet
from aioway.batches import Chunk, Vector
from aioway.tables import ColumnView, SelectView

from .streams import Stream

__all__ = ["StreamColumnView"]


@dcls.dataclass(frozen=True)
class StreamColumnView(Iterator[Vector], ColumnView[Stream]):
    """
    A column reference (on a stream).
    Performs ``__next__`` and yield ``Vector``s.
    """

    column: str
    "The column to select."

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Vector:
        batch = next(self.table)
        return batch[self.column]

    @property
    def attr(self) -> Attr:
        return self.table.attrs[self.column]


@typing.final
@dcls.dataclass(frozen=True)
class StreamSelectView(Stream, SelectView[Stream]):
    """
    The view generated when calling ``Stream.select``.
    """

    COLUMN_TYPE = StreamColumnView

    columns: Sequence[str]
    "The columns to pick up."

    def __iter__(self) -> Self:
        return self

    @typing.override
    def _read(self) -> Chunk:
        batch = next(self.table)
        return batch.select(*self.columns)

    @property
    @typing.override
    def size(self) -> int:
        return self.table.size

    @property
    @typing.override
    def attrs(self) -> AttrSet:
        return AttrSet.from_dict({col: self.table.attrs[col] for col in self.columns})

    @typing.override
    def _children(self) -> Generator[Stream, None, None]:
        yield self.table

    @typing.override
    def select(self, *keys: str):
        for key in keys:
            if key not in self.attrs:
                raise ValueError(f"Key: {key} not in {self.attrs=}.")

        return self.table.select(*keys)
