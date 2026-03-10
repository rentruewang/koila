# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import logging
import typing
from abc import ABC
from collections.abc import KeysView

from aioway._exprs import Expr
from aioway.tables import Table

__all__ = ["SymbolExpr", "ColumnSymbolExpr", "TableSymbolExpr"]

LOGGER = logging.getLogger(__name__)


class SymbolExpr(Expr[str], ABC):
    """
    An (extended) projection operator that can be reprsented as an expression.
    """

    @typing.final
    def __str__(self) -> str:
        return super().compute()

    @typing.override
    @abc.abstractmethod
    def _compute(self) -> str: ...

    @typing.final
    def _return_type(self) -> type[str]:
        return str


class ColumnSymbolExpr(SymbolExpr, ABC):
    @abc.abstractmethod
    def _compute(self) -> str: ...

    @typing.final
    def __not__(self):
        from .ufuncs import NotColExpr

        return NotColExpr(self)

    @typing.final
    def __neg__(self):
        from .ufuncs import NegColExpr

        return NegColExpr(self)

    @typing.final
    def __add__(self, other: "ColumnSymbolExpr"):
        from .ufuncs import AddColExpr

        return AddColExpr(self, other)

    @typing.final
    def __radd__(self, other: "ColumnSymbolExpr"):
        return other + self

    @typing.final
    def __sub__(self, other: "ColumnSymbolExpr"):
        from .ufuncs import SubColExpr

        return SubColExpr(self, other)

    @typing.final
    def __rsub__(self, other: "ColumnSymbolExpr"):
        return other - self

    @typing.final
    def __mul__(self, other: "ColumnSymbolExpr"):
        from .ufuncs import MultColExpr

        return MultColExpr(self, other)

    @typing.final
    def __rmul__(self, other: "ColumnSymbolExpr"):
        return other * self

    @typing.final
    def __truediv__(self, other: "ColumnSymbolExpr"):
        from .ufuncs import TrueDivColExpr

        return TrueDivColExpr(self, other)

    @typing.final
    def __rtruediv__(self, other: "ColumnSymbolExpr"):
        return other / self

    @typing.final
    def __floordiv__(self, other: "ColumnSymbolExpr"):
        from .ufuncs import FloorDivColExpr

        return FloorDivColExpr(self, other)

    @typing.final
    def __rfloordiv__(self, other: "ColumnSymbolExpr"):
        return other // self

    @typing.final
    def __pow__(self, other: "ColumnSymbolExpr"):
        from .ufuncs import ExpColExpr

        return ExpColExpr(self, other)

    @typing.final
    def __rpow__(self, other: "ColumnSymbolExpr"):
        return other**self

    @typing.final
    def __eq__(self, other: object):
        if isinstance(other, ColumnSymbolExpr):
            from .ufuncs import EqColExpr

            return EqColExpr(self, other)

        return NotImplemented

    @typing.final
    def __ne__(self, other: object):
        if isinstance(other, ColumnSymbolExpr):
            from .ufuncs import NeColExpr

            return NeColExpr(self, other)

        return NotImplemented

    @typing.final
    def __gt__(self, other: "ColumnSymbolExpr"):
        from .ufuncs import GtColExpr

        return GtColExpr(self, other)

    @typing.final
    def __ge__(self, other: "ColumnSymbolExpr"):
        from .ufuncs import GeColExpr

        return GeColExpr(self, other)

    @typing.final
    def __lt__(self, other: "ColumnSymbolExpr"):
        from .ufuncs import LtColExpr

        return LtColExpr(self, other)

    @typing.final
    def __le__(self, other: "ColumnSymbolExpr"):
        from .ufuncs import LeColExpr

        return LeColExpr(self, other)


class TableSymbolExpr(SymbolExpr, Table[ColumnSymbolExpr], ABC):
    @abc.abstractmethod
    def _compute(self) -> str: ...

    @abc.abstractmethod
    def keys(self) -> KeysView[str]: ...

    @typing.override
    def column(self, key: str) -> ColumnSymbolExpr:
        from .cols import GetItemExpr

        return GetItemExpr(table=self, column=key)

    @typing.override
    def select(self, *keys: str) -> "TableSymbolExpr":
        from .tables import SelectExpr

        return SelectExpr(table=self, columns=keys)
