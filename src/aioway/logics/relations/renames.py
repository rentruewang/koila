# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
from collections.abc import Mapping
from typing import TypeVar

from aioway.logics.dtypes import Schema

from .nodes import PlanNode
from .relations import Relation, RelationVisitor

_T = TypeVar("_T")
_P = TypeVar("_P", bound=PlanNode)


@dcls.dataclass(frozen=True)
class RenameRelation(Relation[_P]):
    """
    The rename operator in relational algebra, denoted by ρ.

    The rename operation is responsible for 2 things::

        #. Renaming the columns of the previous relation.
        #. Renaming the table itself.
    """

    prev: _P
    """
    The table for which to rename.
    """

    cols: Mapping[str, str]
    """
    The column rename mapping.
    """

    def accept(self, visitor: RelationVisitor[_P, _T]) -> _T:
        return visitor.rename(self)

    @property
    def sources(self) -> tuple[_P]:
        return (self.prev,)

    @property
    def schema(self) -> Schema:
        return Schema.mapping(
            {
                self.cols.get(name, name): dtype
                for name, dtype in self.prev.schema.items()
            }
        )
