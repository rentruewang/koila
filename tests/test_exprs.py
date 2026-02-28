# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
from collections import defaultdict as DefaultDict

import pytest

from aioway.exprs import (
    ColumnExpr,
    ExactColExpr,
    InfixColExpr,
    PrefixColExpr,
    SourceExpr,
)


@dcls.dataclass(frozen=True)
class SymbolicColumn: ...


@dcls.dataclass(frozen=True)
class SymbolicTable:
    cached: dict[str, SymbolicColumn] = dcls.field(
        default_factory=lambda: DefaultDict(SymbolicColumn), init=False
    )

    def __getitem__(self, key: str):
        return self.cached[key]


@pytest.fixture
def a():
    return SourceExpr("a")


@pytest.fixture
def b():
    return SourceExpr("b")


@pytest.fixture
def table_a():
    return SymbolicTable()


@pytest.fixture
def table_b():
    return SymbolicTable()


@pytest.fixture
def c(a):
    return ExactColExpr(a, "c")


@pytest.fixture
def d(a):
    return ExactColExpr(a, "d")


@pytest.fixture
def e(b):
    return ExactColExpr(b, "e")


@pytest.fixture
def column_c(table_a):
    return table_a["c"]


@pytest.fixture
def column_d(table_a):
    return table_a["d"]


@pytest.fixture
def column_e(table_b):
    return table_b["e"]


@pytest.fixture
def tables(table_a, table_b):
    return {"a": table_a, "b": table_b}


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


@pytest.mark.parametrize("op", "+-")
def test_prefix_op_repr(col_expr, op):
    assert str(PrefixColExpr(op=op, child=col_expr)) == f"{op}{col_expr!s}"
