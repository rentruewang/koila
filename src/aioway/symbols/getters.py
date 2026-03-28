# Copyright (c) AIoWay Authors - All Rights Reserved

"The expressions represent a table or a view."

import typing
from collections.abc import KeysView, Sequence

from aioway._typing import SeqKeysView

from . import _common
from .exprs import ColSymbol, TableSymbol

__all__ = ["SourceSymbol", "SelectSymbol", "GetItemSymbol"]


@typing.final
@_common.symbol_dataclass
class SourceSymbol(TableSymbol):
    name: str
    "The table's name. Matches the table names given in the `subs` method."

    columns: Sequence[str]
    "The columns in the table."

    def __str__(self) -> str:
        return self.name

    def keys(self) -> KeysView[str]:
        return SeqKeysView(self.columns)


@_common.symbol_dataclass
class SelectSymbol(TableSymbol):
    table: TableSymbol
    "The source to the selection."

    columns: Sequence[str]
    "The columns to select."

    @typing.override
    def __str__(self) -> str:
        return f"select({self.table!s}, {self.columns!r})"

    @typing.override
    def keys(self) -> KeysView[str]:
        return SeqKeysView(self.columns)


@_common.symbol_dataclass
class GetItemSymbol(ColSymbol):
    "Perform the `__getitem__` operation. Select one of the keys."

    table: TableSymbol
    """
    The table expression that the column would operate on.
    """

    column: str
    """
    The name of the column.
    """

    @typing.override
    def __str__(self) -> str:
        return f"{self.table!s}.{self.column}"
