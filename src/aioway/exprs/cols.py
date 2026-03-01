# Copyright (c) AIoWay Authors - All Rights Reserved

"The expressions representing a column."

import dataclasses as dcls
import typing

from .exprs import ColumnExpr, TableExpr

__all__ = ["GetItemExpr", "PrefixColExpr", "InfixColExpr"]


@dcls.dataclass(frozen=True, eq=False)
class GetItemExpr(ColumnExpr):
    NUM_ARGS = 0

    table: TableExpr
    """
    The table expression that the column would operate on.
    """

    column: str
    """
    The name of the column.
    """

    @typing.override
    def __str__(self) -> str:
        return f"{self.table!s}.{self.column}"

    @typing.override
    def _children(self):
        return
        yield


@dcls.dataclass(frozen=True, eq=False)
class PrefixColExpr(ColumnExpr):
    NUM_ARGS = 1

    op: str
    """
    The operator for the prefix column.
    """

    child: ColumnExpr
    "The child"

    @typing.override
    def __str__(self) -> str:
        return f"{self.op}{self.child}"

    @typing.override
    def _children(self):
        yield self.child


@dcls.dataclass(frozen=True, eq=False)
class InfixColExpr(ColumnExpr):
    NUM_ARGS = 2
    op: str

    left: ColumnExpr
    right: ColumnExpr

    @typing.override
    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"

    @typing.override
    def _children(self):
        yield self.left
        yield self.right
