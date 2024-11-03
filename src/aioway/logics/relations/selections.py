# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
from typing import TypeVar

from aioway.logics.dtypes import Schema
from aioway.logics.exprs import Expr

from .nodes import PlanNode
from .relations import Relation, RelationVisitor

_T = TypeVar("_T", covariant=True)
_P = TypeVar("_P", bound=PlanNode)


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
    def schema(self) -> Schema:
        return self.prev.schema
