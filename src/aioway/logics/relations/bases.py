# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
from typing import TypeVar

from aioway.logics.dtypes import Schema

from .nodes import PlanNode
from .relations import Relation, RelationVisitor

_T = TypeVar("_T")
_P = TypeVar("_P", bound=PlanNode)


@dcls.dataclass(frozen=True)
class BaseRelation(Relation[_P]):
    """
    Base class represents concrete / source data.
    """

    base: Schema
    """
    The schema of the base relation.
    This acts as the source for all the internal nodes.
    """

    filename: str
    """
    The resource locator of this relation.
    """

    def accept(self, visitor: RelationVisitor[_P, _T]) -> _T:
        return visitor.base(self)

    @property
    def sources(self) -> tuple[()]:
        return ()

    @property
    def schema(self) -> Schema:
        return self.base
