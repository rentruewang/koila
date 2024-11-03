# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
from typing import TypeVar

from aioway.logics.dtypes import Schema

from .nodes import PlanNode
from .relations import Relation, RelationVisitor

_T = TypeVar("_T")
_P = TypeVar("_P", bound=PlanNode)


@dcls.dataclass(frozen=True)
class TransformRelation(Relation[_P]):
    """
    The transform operator, unique to ``aioway``, denoted by λ.
    """

    prev: _P
    """
    The input data to transform.
    """

    to: Schema
    """
    The output format.
    """

    def accept(self, visitor: RelationVisitor[_P, _T]) -> _T:
        return visitor.transform(self)

    @property
    def dtype(self) -> Schema:
        return self.to

    @property
    def sources(self) -> tuple[_P]:
        return (self.prev,)

    @property
    def schema(self) -> Schema:
        return self.to
