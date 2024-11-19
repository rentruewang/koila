# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls

from aioway.schemas import TableSchema

from .relations import P, Relation, RelationVisitor, T

__all__ = ["TransformRelation"]


@dcls.dataclass(frozen=True)
class TransformRelation(Relation[P]):
    """
    The transform operator, unique to ``aioway``, denoted by λ.
    """

    prev: P
    """
    The input data to transform.
    """

    to: TableSchema
    """
    The output format.
    """

    def accept(self, visitor: RelationVisitor[P, T]) -> T:
        return visitor.transform(self)

    @property
    def dtype(self) -> TableSchema:
        return self.to

    @property
    def sources(self) -> tuple[P]:
        return (self.prev,)

    @property
    def schema(self) -> TableSchema:
        return self.to
