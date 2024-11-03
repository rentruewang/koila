# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
from typing import TypeVar

from aioway.logics.dtypes import Schema

from .nodes import PlanNode
from .relations import Relation, RelationVisitor

_T = TypeVar("_T")
_P = TypeVar("_P", bound=PlanNode)


@dcls.dataclass(frozen=True)
class UnionRelation(Relation[_P]):
    """
    The union operator in relational algebra, denoted by ∪.

    Union concatenates two relations with the same type.
    """

    top: _P
    """
    The top node to union.
    """

    down: _P
    """
    The top node to union.
    """

    def __post_init__(self) -> None:
        if self.top.schema != self.down.schema:
            raise ValueError("Incompatible tables cannot be merged together.")

    def accept(self, visitor: RelationVisitor[_P, _T]) -> _T:
        return visitor.union(self)

    @property
    def sources(self) -> tuple[_P, _P]:
        return self.top, self.down

    @property
    def schema(self) -> Schema:
        return self.top.schema
