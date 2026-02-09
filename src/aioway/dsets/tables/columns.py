# Copyright (c) AIoWay Authors - All Rights Reserved

"``Column``s are a column of ``Table``."

import dataclasses as dcls
import typing

from aioway.attrs import Attr
from aioway.batches import Vector

if typing.TYPE_CHECKING:
    from .tables import Table, TableIndex

__all__ = ["ColumnRef"]


@dcls.dataclass(frozen=True)
class ColumnRef:
    """
    A column reference.
    Performs ``__getitem__`` on a ``Table``, then select the column.
    """

    table: "Table"
    "The original table which this reference is derived from."

    column: str
    "The column to select."

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: "TableIndex", /) -> Vector:
        batch = self.table[idx]
        return batch[self.column]

    @property
    def attr(self) -> Attr:
        return self.table.attrs[self.column]
