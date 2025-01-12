# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls

from aioway.schemas import Schema

from .relations import Relation, RelationVisitor, RelNode

__all__ = ["TransformRelation"]


@dcls.dataclass(frozen=True)
class TransformRelation[P: RelNode](Relation[P]):
    """
    The transform operator, unique to ``aioway``, denoted by λ.
    """

    prev: P
    """
    The input data to transform.
    """

    to: Schema
    """
    The output format.
    """

    def accept[T](self, visitor: RelationVisitor[P, T]) -> T:
        return visitor.transform(self)

    @property
    def dtype(self) -> Schema:
        return self.to

    @property
    def sources(self) -> tuple[P]:
        return (self.prev,)

    @property
    def schema(self) -> Schema:
        return self.to
