# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls

from aioway.relalg.exprs import Expr
from aioway.schemas import Schema

from .relations import Relation, RelationVisitor, RelNode

__all__ = ["SelectionRelation"]


@dcls.dataclass(frozen=True)
class SelectionRelation[P: RelNode](Relation[P]):
    """
    The selection operator in relational algebra, denoted by σ.
    """

    prev: P
    """
    The input data to filter.
    """

    condition: Expr
    """
    Condition for filtering the inputs.
    """

    def accept[T](self, visitor: RelationVisitor[P, T]) -> T:
        return visitor.select(self)

    @property
    def sources(self) -> tuple[P]:
        return (self.prev,)

    @property
    def schema(self) -> Schema:
        return self.prev.schema
