# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls

from aioway.schemas import TableSchema

from .nodes import RelNode
from .relations import Relation, RelationVisitor

__all__ = ["BaseRelation"]


@dcls.dataclass(frozen=True)
class BaseRelation[P: RelNode](Relation[P]):
    """
    Base class represents concrete / source data.
    """

    base: TableSchema
    """
    The schema of the base relation.
    This acts as the source for all the internal nodes.
    """

    filename: str
    """
    The resource locator of this relation.
    """

    def accept[T](self, visitor: RelationVisitor[P, T]) -> T:
        return visitor.base(self)

    @property
    def sources(self) -> tuple[()]:
        return ()

    @property
    def schema(self) -> TableSchema:
        return self.base
