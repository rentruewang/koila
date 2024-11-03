# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
import enum
from enum import StrEnum
from typing import TypeVar

from aioway.logics.dtypes import Schema

from .nodes import PlanNode
from .relations import Relation, RelationVisitor

_T = TypeVar("_T")
_P = TypeVar("_P", bound=PlanNode)


class Product(StrEnum):
    """
    The different types of the cartisian products.
    """

    INNER = enum.auto()
    """
    The inner product type.
    """

    OUTER = enum.auto()
    """
    The outer product type.
    """

    LEFT = enum.auto()
    """
    Left join type.
    """

    RIGHT = enum.auto()
    """
    Right join type.
    """


@dcls.dataclass(frozen=True)
class ProductRelation(Relation[_P]):
    """
    The cartesian product of two relations in relational algebra, denoted by X.
    """

    left: _P
    """
    The lhs of the product relation.
    """

    right: _P
    """
    The rhs of the product relation.
    """

    keys: tuple[str, str]
    prod: Product

    def accept(self, visitor: RelationVisitor[_P, _T]) -> _T:
        return visitor.product(self)

    @property
    def sources(self) -> tuple[_P, _P]:
        return self.left, self.right

    @property
    def schema(self) -> Schema:
        return self.left.schema | self.right.schema


@dcls.dataclass(frozen=True)
class ConcatRelation(Relation[_P]):
    left: _P
    right: _P

    def accept(self, visitor: RelationVisitor[_P, _T]) -> _T:
        return visitor.concat(self)

    @property
    def sources(self) -> tuple[_P, _P]:
        return self.left, self.right

    @property
    def schema(self) -> Schema:
        return self.left.schema | self.right.schema
