# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
from collections.abc import Mapping

from aioway.schemas import TableSchema

from .relations import P, Relation, RelationVisitor, T

__all__ = ["RenameRelation"]


@dcls.dataclass(frozen=True)
class RenameRelation(Relation[P]):
    """
    The rename operator in relational algebra, denoted by ρ.

    The rename operation is responsible for 2 things::

        #. Renaming the columns of the previous relation.
        #. Renaming the table itself.
    """

    prev: P
    """
    The table for which to rename.
    """

    cols: Mapping[str, str]
    """
    The column rename mapping.
    """

    def accept(self, visitor: RelationVisitor[P, T]) -> T:
        return visitor.rename(self)

    @property
    def sources(self) -> tuple[P]:
        return (self.prev,)

    @property
    def schema(self) -> TableSchema:
        return TableSchema.mapping(
            {
                self.cols.get(name, name): dtype
                for name, dtype in self.prev.schema.items()
            }
        )
