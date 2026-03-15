# Copyright (c) AIoWay Authors - All Rights Reserved

"The expressions represent a table or a view."

import typing
from collections.abc import KeysView, Sequence

from aioway._typing import SeqKeysView

from . import _common
from .exprs import ColSymExpr, TableSymExpr

__all__ = ["SourceExpr", "SelectExpr", "GetItemExpr"]


@typing.final
@_common.symbol_dataclass
class SourceExpr(TableSymExpr):
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


@_common.symbol_dataclass
class SelectExpr(TableSymExpr):
    table: TableSymExpr
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
class GetItemExpr(ColSymExpr):
    "Perform the ``__getitem__`` operation. Select one of the keys."

    table: TableSymExpr
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
