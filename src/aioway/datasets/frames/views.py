# Copyright (c) AIoWay Authors - All Rights Reserved

"`View`s are columns and projections of tables."

import dataclasses as dcls
import typing

from aioway import chunks
from aioway._typing import BatchIndex

from ..datasets import DatasetColumnView, DatasetSelectView
from .frames import Frame, IntArray

__all__ = ["FrameColumnView", "FrameSelectView"]


@dcls.dataclass(frozen=True)
class FrameColumnView(DatasetColumnView[Frame]):
    """
    A column reference to a `Frame`.
    Performs `__getitem__` on a `Frame`, then select the column.
    """

    def __len__(self) -> int:
        return len(self.dset)

    def __getitem__(self, idx: BatchIndex, /) -> chunks.Vector:
        batch = self.dset[idx]
        return batch[self.col]

    @classmethod
    def from_column(cls, dataset: Frame, /, column: str) -> typing.Self:
        return cls(dataset, column)


@dcls.dataclass(frozen=True)
class FrameSelectView(DatasetSelectView[Frame], Frame):
    """
    A selection view on the `Frame`.
    """

    COLUMN_TYPE = FrameColumnView

    @typing.override
    def __len__(self) -> int:
        return len(self.dset)

    @typing.override
    def _getitem(self, idx: IntArray, /) -> chunks.Chunk:
        items = self.dset[idx]
        return items.select(*self.cols)

    @classmethod
    def from_columns(cls, dataset: Frame, /, *columns: str) -> typing.Self:
        return cls(dataset, columns)
