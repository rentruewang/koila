# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import logging
import typing
from abc import ABC
from collections.abc import KeysView

from aioway._exprs import Expr
from aioway._tables import Table

__all__ = ["SymbolExpr", "ColSymExpr", "TableSymExpr"]

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
    def _return_type(self):
        return str


class ColSymExpr(SymbolExpr, ABC):
    @abc.abstractmethod
    def _compute(self) -> str: ...

    @typing.final
    def __invert__(self):
        from .ufuncs import InvColExpr

        return InvColExpr(self)

    @typing.final
    def __neg__(self):
        from .ufuncs import NegColExpr

        return NegColExpr(self)

    @typing.final
    def __add__(self, other: ColSymExpr):
        from .ufuncs import AddColExpr

        return AddColExpr(self, other)

    @typing.final
    def __sub__(self, other: ColSymExpr):
        from .ufuncs import SubColExpr

        return SubColExpr(self, other)

    @typing.final
    def __mul__(self, other: ColSymExpr):
        from .ufuncs import MultColExpr

        return MultColExpr(self, other)

    @typing.final
    def __truediv__(self, other: ColSymExpr):
        from .ufuncs import TrueDivColExpr

        return TrueDivColExpr(self, other)

    @typing.final
    def __floordiv__(self, other: ColSymExpr):
        from .ufuncs import FloorDivColExpr

        return FloorDivColExpr(self, other)

    @typing.final
    def __pow__(self, other: ColSymExpr):
        from .ufuncs import ExpColExpr

        return ExpColExpr(self, other)

    @typing.final
    def __eq__(self, other: object):
        if isinstance(other, ColSymExpr):
            from .ufuncs import EqColExpr

            return EqColExpr(self, other)

        return NotImplemented

    @typing.final
    def __ne__(self, other: object):
        if isinstance(other, ColSymExpr):
            from .ufuncs import NeColExpr

            return NeColExpr(self, other)

        return NotImplemented

    @typing.final
    def __gt__(self, other: ColSymExpr):
        from .ufuncs import GtColExpr

        return GtColExpr(self, other)

    @typing.final
    def __ge__(self, other: ColSymExpr):
        from .ufuncs import GeColExpr

        return GeColExpr(self, other)

    @typing.final
    def __lt__(self, other: ColSymExpr):
        from .ufuncs import LtColExpr

        return LtColExpr(self, other)

    @typing.final
    def __le__(self, other: ColSymExpr):
        from .ufuncs import LeColExpr

        return LeColExpr(self, other)


class TableSymExpr(SymbolExpr, Table[ColSymExpr], ABC):
    @typing.overload
    def __getitem__(self, key: str) -> ColSymExpr: ...

    @typing.overload
    def __getitem__(self, key: list[str]) -> TableSymExpr: ...

    @typing.no_type_check
    def __getitem__(self, key):
        return Table.__getitem__(self, key)

    @abc.abstractmethod
    def _compute(self) -> str: ...

    @abc.abstractmethod
    def keys(self) -> KeysView[str]: ...

    @typing.override
    def column(self, key: str) -> ColSymExpr:
        from .getters import GetItemExpr

        return GetItemExpr(table=self, column=key)

    @typing.override
    def select(self, *keys: str) -> TableSymExpr:
        from .getters import SelectExpr

        return SelectExpr(table=self, columns=keys)
