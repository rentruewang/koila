# Copyright (c) AIoWay Authors - All Rights Reserved

import operator
import typing
from collections import abc as cabc

import pytest

from aioway import symbols


@pytest.fixture
def a() -> symbols.SourceSymbol:
    return symbols.SourceSymbol("a", "cde")


@pytest.fixture
def b() -> symbols.SourceSymbol:
    return symbols.SourceSymbol("b", "cde")


@pytest.fixture
def c(a: symbols.SourceSymbol):
    return a["c"]


@pytest.fixture
def d(a: symbols.SourceSymbol):
    return a["d"]


@pytest.fixture
def e(b: symbols.SourceSymbol):
    return b["e"]


@pytest.fixture(params="cde")
def col_expr(request: pytest.FixtureRequest) -> symbols.ColSymbol:
    return request.getfixturevalue(request.param)


def test_col_repr(c: str, d: str, e: str):
    assert str(c) == "a.c"
    assert str(d) == "a.d"
    assert str(e) == "b.e"


class _OpFunc(typing.NamedTuple):
    name: str
    func: cabc.Callable[..., object]


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
def test_infix_op_repr(
    c: str, d: str, op: str, func: cabc.Callable[[typing.Any, typing.Any], typing.Any]
) -> None:
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
def test_binray_ufunc_repr(
    c: str, d: str, op: str, func: cabc.Callable[[typing.Any, typing.Any], typing.Any]
) -> None:
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
def test_binary_ufunc_type(
    c: str, e: str, op: cabc.Callable[[typing.Any, typing.Any], typing.Any]
) -> None:
    assert isinstance(expr := op(c, e), symbols.ColSymbol), type(expr)
    assert isinstance(expr := op(e, c), symbols.ColSymbol), type(expr)


@pytest.mark.parametrize(
    "op,func",
    [
        _OpFunc("-", operator.neg),
        _OpFunc("~", operator.inv),
    ],
)
def test_prefix_op_repr(
    col_expr: symbols.ColSymbol,
    op: str,
    func: cabc.Callable[[typing.Any, typing.Any], typing.Any],
):
    assert str(func(col_expr)) == f"{op}{col_expr!s}"
