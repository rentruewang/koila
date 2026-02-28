# Copyright (c) AIoWay Authors - All Rights Reserved

"The expressions representing a column."

import dataclasses as dcls
import typing

from aioway.tables import Table

from .exprs import ColumnExpr, TableExpr

__all__ = ["ExactColExpr", "PrefixColExpr", "InfixColExpr"]


@dcls.dataclass(frozen=True)
class ExactColExpr(ColumnExpr):
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
    def subs[C](self, **tables: Table[C]) -> C:
        table = self.table.subs(**tables)
        return table.column(self.column)

    @typing.override
    def __str__(self) -> str:
        return f"{self.table!s}.{self.column}"

    @typing.override
    def _children(self):
        return
        yield


def prefix_op(op: str):
    match op:
        case "+":
            return lambda x: x
        case "-":
            return lambda x: -x
        case _:
            raise NotImplementedError(op)


@dcls.dataclass(frozen=True)
class PrefixColExpr(ColumnExpr):
    NUM_ARGS = 1

    op: str
    """
    The operator for the prefix column.
    """

    child: ColumnExpr
    "The child"

    @typing.override
    def subs[C](self, **tables: Table[C]) -> C:
        child = self.child.subs(**tables)
        op = prefix_op(self.op)
        return op(child)

    @typing.override
    def __str__(self) -> str:
        return f"{self.op}{self.child}"

    @typing.override
    def _children(self):
        yield self.child


def infix_op(op: str):
    match op:
        case "+":
            return lambda l, r: l + r

        case "-":
            return lambda l, r: l - r

        case "*":
            return lambda l, r: l * r

        case "/":
            return lambda l, r: l / r

        case "//":
            return lambda l, r: l // r

        case "**":
            return lambda l, r: l**r

        case "==":
            return lambda l, r: l == r

        case "!=":
            return lambda l, r: l != r

        case ">=":
            return lambda l, r: l >= r

        case "<=":
            return lambda l, r: l <= r

        case ">":
            return lambda l, r: l > r

        case "<":
            return lambda l, r: l < r

        case _:
            raise NotImplementedError


@dcls.dataclass(frozen=True)
class InfixColExpr(ColumnExpr):
    NUM_ARGS = 2
    op: str
    left: ColumnExpr
    right: ColumnExpr

    @typing.override
    def subs[C](self, **tables: Table[C]) -> C:
        left = self.left.subs(**tables)
        right = self.right.subs(**tables)

        op = infix_op(self.op)

        return op(left, right)

    @typing.override
    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"

    @typing.override
    def _children(self):
        yield self.left
        yield self.right
