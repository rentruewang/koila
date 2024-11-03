# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
from typing import TypeVar

from aioway.logics.dtypes import Schema
from aioway.logics.exprs import Expr

from .nodes import PlanNode
from .relations import Relation

_T = TypeVar("_T", covariant=True)


@dcls.dataclass(frozen=True)
class SelectionRelation(Relation):
    """
    The selection operator in relational algebra, denoted by σ.
    """

    prev: PlanNode
    """
    The input data to filter.
    """

    condition: Expr
    """
    Condition for filtering the inputs.
    """

    def accept(self, visitor: Relation.Visitor[_T]) -> _T:
        return visitor.select(self)

    @property
    def sources(self) -> tuple[PlanNode]:
        return (self.prev,)

    @property
    def schema(self) -> Schema:
        return self.prev.schema
