# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
from typing import TypeVar

from aioway.logics.dtypes import Schema

from .nodes import PlanNode
from .relations import Relation

_T = TypeVar("_T")


@dcls.dataclass(frozen=True)
class ViewRelation(Relation):
    """
    View node represents a terminal node in the relations.
    """

    prev: PlanNode
    """
    The node for which to perform viewing (most likely IO operation).
    """

    unique: str
    """
    The name of the terminal resource location. Maybe a file.
    """

    def accept(self, visitor: Relation.Visitor[_T]) -> _T:
        return visitor.view(self)

    @property
    def sources(self) -> tuple[PlanNode]:
        return (self.prev,)

    @property
    def schema(self) -> Schema:
        return self.prev.schema
