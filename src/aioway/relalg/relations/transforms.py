# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
from typing import TypeVar

from aioway.schemas import TableSchema

from .nodes import RelNode
from .relations import Relation, RelationVisitor

_T = TypeVar("_T")
_P = TypeVar("_P", bound=RelNode)


@dcls.dataclass(frozen=True)
class TransformRelation(Relation[_P]):
    """
    The transform operator, unique to ``aioway``, denoted by λ.
    """

    prev: _P
    """
    The input data to transform.
    """

    to: TableSchema
    """
    The output format.
    """

    def accept(self, visitor: RelationVisitor[_P, _T]) -> _T:
        return visitor.transform(self)

    @property
    def dtype(self) -> TableSchema:
        return self.to

    @property
    def sources(self) -> tuple[_P]:
        return (self.prev,)

    @property
    def schema(self) -> TableSchema:
        return self.to
