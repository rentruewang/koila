# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
from typing import TypeVar

from aioway.relalg.exprs import Expr
from aioway.schemas import TableSchema

from .nodes import RelNode
from .relations import Relation, RelationVisitor

_T = TypeVar("_T", covariant=True)
_P = TypeVar("_P", bound=RelNode)


@dcls.dataclass(frozen=True)
class SelectionRelation(Relation[_P]):
    """
    The selection operator in relational algebra, denoted by σ.
    """

    prev: _P
    """
    The input data to filter.
    """

    condition: Expr
    """
    Condition for filtering the inputs.
    """

    def accept(self, visitor: RelationVisitor[_P, _T]) -> _T:
        return visitor.select(self)

    @property
    def sources(self) -> tuple[_P]:
        return (self.prev,)

    @property
    def schema(self) -> TableSchema:
        return self.prev.schema
