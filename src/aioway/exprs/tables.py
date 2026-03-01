# Copyright (c) AIoWay Authors - All Rights Reserved

"The expressions represent a table or a view."

import dataclasses as dcls
import typing
from collections.abc import Iterator, KeysView, Sequence

from .exprs import Expr, TableExpr

__all__ = ["SourceExpr", "SelectExpr"]


@typing.final
@dcls.dataclass(frozen=True)
class SourceExpr(TableExpr):
    NUM_ARGS = 0

    name: str
    "The table's name. Matches the table names given in the ``subs`` method."

    columns: Sequence[str]

    @typing.override
    def __str__(self) -> str:
        return self.name

    def keys(self) -> KeysView[str]:
        return SeqKeysView(self.columns)

    @typing.override
    def _children(self) -> Iterator[Expr]:
        return
        yield


@typing.final
@dcls.dataclass(frozen=True)
class SelectExpr(TableExpr):
    NUM_ARGS = 1

    table: TableExpr
    "The source to the selection."

    columns: Sequence[str]
    "The columns to select."

    @typing.override
    def __str__(self) -> str:
        return f"select({self.table!s}, {self.columns!r})"

    @typing.override
    def keys(self) -> KeysView[str]:
        return SeqKeysView(self.columns)

    @typing.override
    def _children(self) -> Iterator[Expr]:
        yield self.table


@dcls.dataclass(frozen=True)
class SeqKeysView(KeysView[str]):
    seq: Sequence[str]

    @typing.override
    def __contains__(self, key: object) -> bool:
        return key in self.seq

    @typing.override
    def __iter__(self):
        yield from self.seq
