# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
import functools
from abc import ABC
from typing import TypeVar

from aioway.backend.volatile import Block

from .tables import Table

_T = TypeVar("_T")


@dcls.dataclass(frozen=True)
class CachedTable(Table, ABC):
    """
    ``CachedTable`` caches the input of a table.
    This table is used whenever an output is shared.
    """

    table: Table
    """
    The input whose ``Block`` computation would be cached.
    """

    @functools.cache
    def __call__(self) -> Block:
        return self.table()

    def accept(self, visitor: Table.Visitor[_T]) -> _T:
        return visitor.cache(self)

    @property
    def sources(self) -> tuple[Table]:
        return (self.table,)
