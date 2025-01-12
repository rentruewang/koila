# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls

from aioway.schemas import Schema

from .relations import Relation, RelationVisitor, RelNode

__all__ = ["ViewRelation"]


@dcls.dataclass(frozen=True)
class ViewRelation[P: RelNode](Relation[P]):
    """
    View node represents a terminal node in the relations.
    """

    prev: P
    """
    The node for which to perform viewing (most likely IO operation).
    """

    def accept[T](self, visitor: RelationVisitor[P, T]) -> T:
        return visitor.view(self)

    @property
    def sources(self) -> tuple[P]:
        return (self.prev,)

    @property
    def schema(self) -> Schema:
        return self.prev.schema
