# Copyright (c) AIoWay Authors - All Rights Reserved

"`View`s are columns and projections of tables."

import dataclasses as dcls
import typing

from aioway import _typing, chunks

from .. import datasets
from . import frames

__all__ = ["FrameColumnView", "FrameSelectView"]


@dcls.dataclass(frozen=True)
class FrameColumnView(datasets.DatasetColumnView[frames.Frame]):
    """
    A column reference to a `frames.Frame`.
    Performs `__getitem__` on a `frames.Frame`, then select the column.
    """

    def __len__(self) -> int:
        return len(self.dset)

    def __getitem__(self, idx: _typing.BatchIndex, /) -> chunks.Vector:
        batch = self.dset[idx]
        return batch[self.col]

    @classmethod
    def from_column(cls, dataset: frames.Frame, /, column: str) -> typing.Self:
        return cls(dataset, column)


@dcls.dataclass(frozen=True)
class FrameSelectView(datasets.DatasetSelectView[frames.Frame], frames.Frame):
    """
    A selection view on the `frames.Frame`.
    """

    COLUMN_TYPE = FrameColumnView

    @typing.override
    def __len__(self) -> int:
        return len(self.dset)

    @typing.override
    def _getitem(self, idx: _typing.IntArray, /) -> chunks.Chunk:
        items = self.dset[idx]
        return items.select(*self.cols)

    @classmethod
    def from_columns(cls, dataset: frames.Frame, /, *columns: str) -> typing.Self:
        return cls(dataset, columns)
