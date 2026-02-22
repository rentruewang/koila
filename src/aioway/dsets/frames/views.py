# Copyright (c) AIoWay Authors - All Rights Reserved

"``Column``s are a column of ``Frame``."

import dataclasses as dcls
import typing
from collections.abc import Sequence

from aioway.attrs import Attr, AttrSet
from aioway.batches import Chunk, Vector
from aioway.tables import ColumnView, SelectView

from .frames import Frame, FrameBatchIndex, IntArray

__all__ = ["FrameColumnView"]


@dcls.dataclass(frozen=True)
class FrameColumnView(ColumnView[Frame]):
    """
    A column reference to a ``Frame``.
    Performs ``__getitem__`` on a ``Frame``, then select the column.
    """

    column: str
    "The column to select."

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: FrameBatchIndex, /) -> Vector:
        batch = self.table[idx]
        return batch[self.column]

    @property
    def attr(self) -> Attr:
        return self.table.attrs[self.column]


@dcls.dataclass(frozen=True)
class FrameSelectView(Frame, SelectView[Frame]):
    """
    A selection view on the ``Frame``.
    """

    COLUMN_TYPE = FrameColumnView

    columns: Sequence[str]
    "The sequence of strings that this ``Frame`` must satisfy."

    @typing.override
    def __len__(self) -> int:
        return len(self.table)

    @typing.override
    def _getitem(self, idx: IntArray, /) -> Chunk:
        items = self.table[idx]
        return items.select(*self.columns)

    @property
    @typing.override
    def attrs(self) -> AttrSet:
        return AttrSet.from_dict({col: self.table.attrs[col] for col in self.columns})
