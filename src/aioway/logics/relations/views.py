# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
from typing import TypeVar

from aioway.logics.dtypes import Schema

from .nodes import PlanNode
from .relations import Relation, RelationVisitor

_T = TypeVar("_T")
_P = TypeVar("_P", bound=PlanNode)


@dcls.dataclass(frozen=True)
class ViewRelation(Relation[_P]):
    """
    View node represents a terminal node in the relations.
    """

    prev: _P
    """
    The node for which to perform viewing (most likely IO operation).
    """

    def accept(self, visitor: RelationVisitor[_P, _T]) -> _T:
        return visitor.view(self)

    @property
    def sources(self) -> tuple[_P]:
        return (self.prev,)

    @property
    def schema(self) -> Schema:
        return self.prev.schema
