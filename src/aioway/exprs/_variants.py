# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing

from aioway import variants

from .exprs import ColumnExpr
from .ufuncs import (
    AddColExpr,
    ColumnExpr,
    EqColExpr,
    ExpColExpr,
    FloorDivColExpr,
    GeColExpr,
    GtColExpr,
    LeColExpr,
    LtColExpr,
    MultColExpr,
    NeColExpr,
    NegColExpr,
    NotColExpr,
    SubColExpr,
    TrueDivColExpr,
)


def unary_op_expr():
    yield "neg", NegColExpr
    yield "not", NotColExpr


def binary_op_expr():
    yield "add", AddColExpr
    yield "sub", SubColExpr
    yield "mul", MultColExpr
    yield "truediv", TrueDivColExpr
    yield "floordiv", FloorDivColExpr
    yield "pow", ExpColExpr
    yield "eq", EqColExpr
    yield "ne", NeColExpr
    yield "ge", GeColExpr
    yield "gt", GtColExpr
    yield "le", LeColExpr
    yield "lt", LtColExpr


@dcls.dataclass(frozen=True)
class ColExprClosure[T]:
    """
    Function to store the column expression class.

    This is created because how python closure works; if we just use normal closures,
    it wouldn't work because we would be enclosing over a loop variable,
    and every function would point to the last iterated loop variable.
    """

    expr_cls: type[T]

    def __str__(self) -> str:
        return self.expr_cls.__name__


def registry_closure():
    for unary_op, unary_expr in unary_op_expr():
        yield unary_op, unary_expr, variants.register_1

    for binary_op, binary_expr in binary_op_expr():
        yield binary_op, binary_expr, variants.register_2


@typing.no_type_check
def register_expr_varants():
    for op_name, expr_func, register in registry_closure():
        register(ColumnExpr, op_name)(expr_func)


register_expr_varants()
