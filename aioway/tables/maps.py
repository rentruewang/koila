# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls

from .execs import UnaryExec
from .tables import Table, TableVisitor

__all__ = ["MapTable"]


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

    def __iter__(self):
        for block in self.table:
            yield self.executor(block)

    def accept[T](self, visitor: TableVisitor[T]) -> T:
        return visitor.map(self)

    @property
    def sources(self) -> tuple[Table]:
        return (self.table,)
