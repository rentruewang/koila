# Copyright (c) AIoWay Authors - All Rights Reserved

"The expressions represent a table or a view."

import dataclasses as dcls
import typing
from collections.abc import Iterator, Sequence

from aioway.tables import Column, Table

from .exprs import Expr, TableExpr

__all__ = ["SourceExpr", "SelectExpr"]


@typing.final
@dcls.dataclass(frozen=True)
class SourceExpr(TableExpr):
    NUM_ARGS = 0

    name: str
    "The table's name. Matches the table names given in the ``subs`` method."

    @typing.override
    def __str__(self) -> str:
        return self.name

    @typing.override
    def subs[C: Column](self, **table: Table[C]) -> Table[C]:
        if self.name in table:
            return table[self.name]

        raise KeyError(f"The table '{self.name}' is not provided.")

    @typing.override
    def _children(self) -> Iterator[Expr]:
        return
        yield


@typing.final
@dcls.dataclass(frozen=True)
class SelectExpr(TableExpr):
    NUM_ARGS = 1

    source: TableExpr
    "The source to the selection."

    columns: Sequence[str]
    "The columns to select."

    @typing.override
    def __str__(self) -> str:
        return f"select({self.source!s}, {self.columns!r})"

    @typing.override
    def subs[C: Column](self, **tables: Table[C]) -> Table[C]:
        source_table = self.source.subs(**tables)
        return source_table.select(*self.columns)

    @typing.override
    def _children(self) -> Iterator[Expr]:
        yield self.source
