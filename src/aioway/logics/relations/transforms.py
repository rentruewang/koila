# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
from collections.abc import Mapping
from typing import TypeVar

from aioway.logics.dtypes import DataType, Schema

from .nodes import PlanNode
from .relations import Relation

_T = TypeVar("_T")


@dcls.dataclass(frozen=True)
class TransformRelation(Relation):
    """
    The transform operator, unique to ``aioway``, denoted by λ.
    """

    prev: PlanNode
    """
    The input data to transform.
    """

    to: Mapping[str, DataType]
    """
    The output format.
    """

    def accept(self, visitor: Relation.Visitor[_T]) -> _T:
        return visitor.transform(self)

    @property
    def dtype(self) -> Mapping[str, DataType]:
        return self.to

    @property
    def sources(self) -> tuple[PlanNode]:
        return (self.prev,)

    @property
    def schema(self) -> Schema:
        return Schema.mapping(self.to)
