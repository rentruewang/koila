# Copyright (c) AIoWay Authors - All Rights Reserved

"The expressions representing a column."

import typing

from . import _common
from .exprs import ColumnSymbolExpr, TableSymbolExpr

__all__ = ["GetItemExpr", "PrefixColExpr", "InfixColExpr"]


@_common.symbol_dataclass
class GetItemExpr(ColumnSymbolExpr):
    table: TableSymbolExpr
    """
    The table expression that the column would operate on.
    """

    column: str
    """
    The name of the column.
    """

    @typing.override
    def _compute(self) -> str:
        return f"{self.table!s}.{self.column}"

    def _inputs(self):
        return (self.table,)


@_common.symbol_dataclass
class PrefixColExpr(ColumnSymbolExpr):
    NUM_ARGS = 1

    op: str
    """
    The operator for the prefix column.
    """

    child: ColumnSymbolExpr
    "The child"

    @typing.override
    def _compute(self) -> str:
        return f"{self.op}{self.child}"

    def _inputs(self):
        return (self.child,)


@_common.symbol_dataclass
class InfixColExpr(ColumnSymbolExpr):
    NUM_ARGS = 2
    op: str

    left: ColumnSymbolExpr
    right: ColumnSymbolExpr

    @typing.override
    def _compute(self) -> str:
        return f"({self.left} {self.op} {self.right})"

    @typing.override
    def _inputs(self):
        return self.left, self.right
