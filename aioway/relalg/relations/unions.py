# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls

from aioway.schemas import TableSchema

from .relations import P, Relation, RelationVisitor, T

__all__ = ["UnionRelation"]


@dcls.dataclass(frozen=True)
class UnionRelation(Relation[P]):
    """
    The union operator in relational algebra, denoted by ∪.

    Union concatenates two relations with the same type.
    """

    top: P
    """
    The top node to union.
    """

    down: P
    """
    The top node to union.
    """

    def __post_init__(self) -> None:
        if self.top.schema != self.down.schema:
            raise ValueError("Incompatible tables cannot be merged together.")

    def accept(self, visitor: RelationVisitor[P, T]) -> T:
        return visitor.union(self)

    @property
    def sources(self) -> tuple[P, P]:
        return self.top, self.down

    @property
    def schema(self) -> TableSchema:
        return self.top.schema
