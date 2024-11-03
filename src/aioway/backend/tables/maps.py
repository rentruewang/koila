# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
from typing import TypeVar

from aioway.backend.volatile import Block, UnaryExec

from .tables import Table

_T = TypeVar("_T")


@dcls.dataclass(frozen=True)
class MapTable(Table):
    """
    ``MapTable`` is a ``Table`` that depends on 1 ``Table``, and apply some transformation.
    For example, renames, projections etc in relational algebra has a single source,
    and can be implemented with ``UnaryTable``.
    """

    executor: UnaryExec
    "The transformation from the input."

    table: Table
    "The source of the current ``Table``."

    def __call__(self) -> Block:
        block = self.table()
        return self.executor(block)

    def accept(self, visitor: Table.Visitor[_T]) -> _T:
        return visitor.map(self)

    @property
    def sources(self) -> tuple[Table]:
        return (self.table,)
