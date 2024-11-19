# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
import enum
from enum import StrEnum

from aioway.schemas import TableSchema

from .relations import P, Relation, RelationVisitor, T

__all__ = ["ConcatRelation", "Product", "ProductRelation"]


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
class ProductRelation(Relation[P]):
    """
    The cartesian product of two relations in relational algebra, denoted by X.
    """

    left: P
    """
    The lhs of the product relation.
    """

    right: P
    """
    The rhs of the product relation.
    """

    keys: tuple[str, str]
    prod: Product

    def accept(self, visitor: RelationVisitor[P, T]) -> T:
        return visitor.product(self)

    @property
    def sources(self) -> tuple[P, P]:
        return self.left, self.right

    @property
    def schema(self) -> TableSchema:
        return self.left.schema | self.right.schema


@dcls.dataclass(frozen=True)
class ConcatRelation(Relation[P]):
    left: P
    right: P

    def accept(self, visitor: RelationVisitor[P, T]) -> T:
        return visitor.concat(self)

    @property
    def sources(self) -> tuple[P, P]:
        return self.left, self.right

    @property
    def schema(self) -> TableSchema:
        return self.left.schema | self.right.schema
