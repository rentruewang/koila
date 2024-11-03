# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
from typing import TypeVar

from aioway.logics.dtypes import Schema

from .relations import Relation

_T = TypeVar("_T")


@dcls.dataclass(frozen=True)
class BaseRelation(Relation):
    """
    Base class represents concrete / source data.
    """

    base: Schema
    """
    The schema of the base relation.
    This acts as the source for all the internal nodes.
    """

    def accept(self, visitor: Relation.Visitor[_T]) -> _T:
        return visitor.base(self)

    @property
    def sources(self) -> tuple[()]:
        return ()

    @property
    def schema(self) -> Schema:
        return self.base
