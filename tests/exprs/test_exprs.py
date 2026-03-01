# Copyright (c) AIoWay Authors - All Rights Reserved


import operator
from collections.abc import Callable
from typing import NamedTuple

import pytest

from aioway.exprs import (
    ColumnExpr,
    GetItemExpr,
    InfixColExpr,
    PrefixColExpr,
    SourceExpr,
)


@pytest.fixture
def a():
    return SourceExpr("a", "cde")


@pytest.fixture
def b():
    return SourceExpr("b", "cde")


@pytest.fixture
def c(a):
    return GetItemExpr(a, "c")


@pytest.fixture
def d(a):
    return GetItemExpr(a, "d")


@pytest.fixture
def e(b):
    return GetItemExpr(b, "e")


@pytest.fixture(params="cde")
def col_expr(request) -> ColumnExpr:
    return request.getfixturevalue(request.param)


def test_col_repr(c, d, e):
    assert str(c) == "a.c"
    assert str(d) == "a.d"
    assert str(e) == "b.e"


@pytest.mark.parametrize("op", ["+", "-", "*", "/", "//", "**"])
def test_infix_op_repr(c, d, op) -> None:
    assert str(InfixColExpr(op=op, left=c, right=d)) == f"({c!s} {op} {d!s})"


class _OpFunc(NamedTuple):
    name: str
    func: Callable[..., object]


@pytest.mark.parametrize(
    "op,func",
    [
        _OpFunc("+", operator.add),
        _OpFunc("-", operator.sub),
        _OpFunc("*", operator.mul),
        _OpFunc("/", operator.truediv),
        _OpFunc("//", operator.floordiv),
        _OpFunc("**", operator.pow),
        _OpFunc("==", operator.eq),
        _OpFunc("!=", operator.ne),
        _OpFunc(">=", operator.ge),
        _OpFunc(">", operator.gt),
        _OpFunc("<=", operator.le),
        _OpFunc("<", operator.lt),
    ],
)
def test_binray_ufunc_repr(c, d, op, func) -> None:
    assert str(func(c, d)) == f"{c!s} {op} {d!s}"
    assert str(func(d, c)) == f"{d!s} {op} {c!s}"


@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        operator.floordiv,
        operator.pow,
        operator.eq,
        operator.ne,
        operator.gt,
        operator.ge,
        operator.lt,
        operator.le,
    ],
)
def test_binary_ufunc_type(c, e, op) -> None:
    assert isinstance(expr := op(c, e), ColumnExpr), type(expr)
    assert isinstance(expr := op(e, c), ColumnExpr), type(expr)


@pytest.mark.parametrize("op", "+-")
def test_prefix_op_repr(col_expr, op):
    assert str(PrefixColExpr(op=op, child=col_expr)) == f"{op}{col_expr!s}"
