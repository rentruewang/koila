# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls

from aioway.schemas import TableSchema

from .relations import P, Relation, RelationVisitor, T

__all__ = ["ViewRelation"]


@dcls.dataclass(frozen=True)
class ViewRelation(Relation[P]):
    """
    View node represents a terminal node in the relations.
    """

    prev: P
    """
    The node for which to perform viewing (most likely IO operation).
    """

    def accept(self, visitor: RelationVisitor[P, T]) -> T:
        return visitor.view(self)

    @property
    def sources(self) -> tuple[P]:
        return (self.prev,)

    @property
    def schema(self) -> TableSchema:
        return self.prev.schema
