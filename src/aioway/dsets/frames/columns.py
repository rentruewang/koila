# Copyright (c) AIoWay Authors - All Rights Reserved

"``Column``s are a column of ``Frame``."

import dataclasses as dcls
import typing

from aioway.attrs import Attr
from aioway.batches import Vector

if typing.TYPE_CHECKING:
    from .frames import Frame, FrameBatchIndex

__all__ = ["FrameColumn"]


@dcls.dataclass(frozen=True)
class FrameColumn:
    """
    A column reference to a ``Frame``.
    Performs ``__getitem__`` on a ``Frame``, then select the column.
    """

    frame: "Frame"
    "The original ``Frame`` which this reference is derived from."

    column: str
    "The column to select."

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: "FrameBatchIndex", /) -> Vector:
        batch = self.frame[idx]
        return batch[self.column]

    @property
    def attr(self) -> Attr:
        return self.frame.attrs[self.column]
