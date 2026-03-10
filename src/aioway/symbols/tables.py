# Copyright (c) AIoWay Authors - All Rights Reserved

"The expressions represent a table or a view."

import typing
from collections.abc import KeysView, Sequence

from . import _common
from .exprs import TableSymbolExpr

__all__ = ["SourceExpr", "SelectExpr"]


@typing.final
@_common.symbol_dataclass
class SourceExpr(TableSymbolExpr):
    NUM_ARGS = 0

    name: str
    "The table's name. Matches the table names given in the ``subs`` method."

    columns: Sequence[str]
    "The columns in the table."

    def _compute(self) -> str:
        return self.name

    def keys(self) -> KeysView[str]:
        return SeqKeysView(self.columns)

    def _inputs(self) -> tuple[()]:
        return ()


@typing.final
@_common.symbol_dataclass
class SelectExpr(TableSymbolExpr):
    NUM_ARGS = 1

    table: TableSymbolExpr
    "The source to the selection."

    columns: Sequence[str]
    "The columns to select."

    @typing.override
    def _compute(self) -> str:
        return f"select({self.table!s}, {self.columns!r})"

    @typing.override
    def keys(self) -> KeysView[str]:
        return SeqKeysView(self.columns)

    @typing.override
    def _inputs(self):
        return (self.table,)


@_common.symbol_dataclass
class SeqKeysView(KeysView[str]):
    seq: Sequence[str]

    @typing.override
    def __contains__(self, key: object) -> bool:
        return key in self.seq

    @typing.override
    def __iter__(self):
        yield from self.seq
