# Copyright (c) AIoWay Authors - All Rights Reserved

"``Column``s are a column of ``Table``."

import dataclasses as dcls
import typing
from typing import Self

from aioway.attrs import Attr
from aioway.batches import Vector

if typing.TYPE_CHECKING:
    from .streams import Stream

__all__ = ["SeriesRef"]


@dcls.dataclass(frozen=True)
class SeriesRef:
    """
    A column reference (on a stream).
    Performs ``__next__`` and yield ``Vector``s.
    """

    stream: "Stream"
    "The original table which this reference is derived from."

    column: str
    "The column to select."

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Vector:
        batch = next(self.stream)
        return batch[self.column]

    @property
    def attr(self) -> Attr:
        return self.stream.attrs[self.column]
