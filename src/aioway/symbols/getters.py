# Copyright (c) AIoWay Authors - All Rights Reserved

"The expressions represent a table or a view."

import typing
from collections import abc as cabc

from aioway._typing import SeqKeysView

from . import _common, exprs

__all__ = ["SourceSymbol", "SelectSymbol", "GetItemSymbol"]


@typing.final
@_common.symbol_dataclass
class SourceSymbol(exprs.TableSymbol):
    name: str
    "The table's name. Matches the table names given in the `subs` method."

    columns: cabc.Sequence[str]
    "The columns in the table."

    def __str__(self) -> str:
        return self.name

    def keys(self) -> cabc.KeysView[str]:
        return SeqKeysView(self.columns)


@_common.symbol_dataclass
class SelectSymbol(exprs.TableSymbol):
    table: exprs.TableSymbol
    "The source to the selection."

    columns: cabc.Sequence[str]
    "The columns to select."

    @typing.override
    def __str__(self) -> str:
        return f"select({self.table!s}, {self.columns!r})"

    @typing.override
    def keys(self) -> cabc.KeysView[str]:
        return SeqKeysView(self.columns)


@_common.symbol_dataclass
class GetItemSymbol(exprs.ColSymbol):
    "Perform the `__getitem__` operation. Select one of the keys."

    table: exprs.TableSymbol
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
