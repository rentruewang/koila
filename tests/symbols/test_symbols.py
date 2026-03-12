# Copyright (c) AIoWay Authors - All Rights Reserved

import operator
from collections.abc import Callable
from typing import NamedTuple

import pytest

from aioway.symbols import (
    ColSymExpr,
    SourceExpr,
)


@pytest.fixture
def a():
    return SourceExpr("a", "cde")


@pytest.fixture
def b():
    return SourceExpr("b", "cde")


@pytest.fixture
def c(a: SourceExpr):
    return a["c"]


@pytest.fixture
def d(a: SourceExpr):
    return a["d"]


@pytest.fixture
def e(b: SourceExpr):
    return b["e"]


@pytest.fixture(params="cde")
def col_expr(request) -> ColSymExpr:
    return request.getfixturevalue(request.param)


def test_col_repr(c, d, e):
    assert str(c) == "a.c"
    assert str(d) == "a.d"
    assert str(e) == "b.e"


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
    ],
)
def test_infix_op_repr(c, d, op, func) -> None:
    assert str(func(c, d)) == f"{c!s} {op} {d!s}"


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
    assert isinstance(expr := op(c, e), ColSymExpr), type(expr)
    assert isinstance(expr := op(e, c), ColSymExpr), type(expr)


@pytest.mark.parametrize(
    "op,func",
    [
        _OpFunc("-", operator.neg),
        _OpFunc("~", operator.inv),
    ],
)
def test_prefix_op_repr(col_expr: ColSymExpr, op, func):
    assert str(func(col_expr)) == f"{op}{col_expr!s}"
