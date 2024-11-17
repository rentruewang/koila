# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
from typing import TypeVar

from aioway.schemas import TableSchema

from .nodes import RelNode
from .relations import Relation, RelationVisitor

_T = TypeVar("_T")
_P = TypeVar("_P", bound=RelNode)


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
    def schema(self) -> TableSchema:
        return self.prev.schema
