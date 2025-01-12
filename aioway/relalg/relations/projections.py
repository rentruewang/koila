# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
from collections.abc import Sequence

from aioway.schemas import Schema

from .relations import Relation, RelationVisitor, RelNode

__all__ = ["ProjectionRelation"]


@dcls.dataclass(frozen=True)
class ProjectionRelation[P: RelNode](Relation[P]):
    """
    The projection operator in relational algebra, denoted by π.

    Todo:
        Verification of this node containing only the subset of what exists in previous nodes.
        This means somehow this has to be associated with a resolver,
        which adds a ``Resolver`` dependency to the ``Relation`` nodes.
        Unsure if that is the right call.
    """

    prev: P
    """
    The table for which to project.
    """

    columns: Sequence[str]
    """
    Columns to select. Must be a subset of input columns.
    """

    def __post_init__(self) -> None:
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("Columns must be unique.")

    def accept[T](self, visitor: RelationVisitor[P, T]) -> T:
        return visitor.project(self)

    @property
    def sources(self) -> tuple[P]:
        return (self.prev,)

    @property
    def schema(self) -> Schema:
        return Schema.mapping({col: self.prev.schema[col] for col in self.columns})
