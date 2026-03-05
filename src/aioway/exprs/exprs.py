# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import logging
import typing
from abc import ABC
from collections.abc import Iterator, KeysView, Sequence
from typing import ClassVar, Literal

from aioway.tables import Table

__all__ = ["Expr", "ColumnExpr", "TableExpr"]

LOGGER = logging.getLogger(__name__)


class Expr(ABC):
    """
    An (extended) projection operator that can be reprsented as an expression.
    """

    NUM_ARGS: ClassVar[Literal[0, 1, 2]]
    """
    The number of inputs of the current operator.
    Should be either 0, 1, 2.
    """

    @abc.abstractmethod
    def __str__(self) -> str:
        "The operator must have a custom representation."

        ...

    def children(self) -> Sequence["Expr"]:
        children = list(self._children())

        if len(children) != self.NUM_ARGS:
            raise AssertionError(
                f"The number of children: {len(children)} does not match "
                f"the number of inputs specified: {self.NUM_ARGS=}"
            )

        return children

    @abc.abstractmethod
    def _children(self) -> Iterator["Expr"]:
        "The children of the current ``Expr``. ``len`` must match ``self.NUM_ARGS``."

        ...


class ColumnExpr(Expr, ABC):
    @abc.abstractmethod
    def __str__(self) -> str: ...

    @typing.final
    def __not__(self):
        from .ufuncs import NotColExpr

        return NotColExpr(self)

    @typing.final
    def __neg__(self):
        from .ufuncs import NegColExpr

        return NegColExpr(self)

    @typing.final
    def __add__(self, other: "ColumnExpr"):
        from .ufuncs import AddColExpr

        return AddColExpr(self, other)

    @typing.final
    def __radd__(self, other: "ColumnExpr"):
        return other + self

    @typing.final
    def __sub__(self, other: "ColumnExpr"):
        from .ufuncs import SubColExpr

        return SubColExpr(self, other)

    @typing.final
    def __rsub__(self, other: "ColumnExpr"):
        return other - self

    @typing.final
    def __mul__(self, other: "ColumnExpr"):
        from .ufuncs import MultColExpr

        return MultColExpr(self, other)

    @typing.final
    def __rmul__(self, other: "ColumnExpr"):
        return other * self

    @typing.final
    def __truediv__(self, other: "ColumnExpr"):
        from .ufuncs import TrueDivColExpr

        return TrueDivColExpr(self, other)

    @typing.final
    def __rtruediv__(self, other: "ColumnExpr"):
        return other / self

    @typing.final
    def __floordiv__(self, other: "ColumnExpr"):
        from .ufuncs import FloorDivColExpr

        return FloorDivColExpr(self, other)

    @typing.final
    def __rfloordiv__(self, other: "ColumnExpr"):
        return other // self

    @typing.final
    def __pow__(self, other: "ColumnExpr"):
        from .ufuncs import ExpColExpr

        return ExpColExpr(self, other)

    @typing.final
    def __rpow__(self, other: "ColumnExpr"):
        return other**self

    @typing.final
    def __eq__(self, other: object):
        if isinstance(other, ColumnExpr):
            from .ufuncs import EqColExpr

            return EqColExpr(self, other)

        return NotImplemented

    @typing.final
    def __ne__(self, other: object):
        if isinstance(other, ColumnExpr):
            from .ufuncs import NeColExpr

            return NeColExpr(self, other)

        return NotImplemented

    @typing.final
    def __gt__(self, other: "ColumnExpr"):
        from .ufuncs import GtColExpr

        return GtColExpr(self, other)

    @typing.final
    def __ge__(self, other: "ColumnExpr"):
        from .ufuncs import GeColExpr

        return GeColExpr(self, other)

    @typing.final
    def __lt__(self, other: "ColumnExpr"):
        from .ufuncs import LtColExpr

        return LtColExpr(self, other)

    @typing.final
    def __le__(self, other: "ColumnExpr"):
        from .ufuncs import LeColExpr

        return LeColExpr(self, other)


class TableExpr(Expr, Table[ColumnExpr], ABC):
    @abc.abstractmethod
    def __str__(self) -> str: ...

    @abc.abstractmethod
    def keys(self) -> KeysView[str]: ...

    @typing.override
    def column(self, key: str) -> ColumnExpr:
        from .cols import GetItemExpr

        return GetItemExpr(table=self, column=key)

    @typing.override
    def select(self, *keys: str) -> "TableExpr":
        from .tables import SelectExpr

        return SelectExpr(table=self, columns=keys)
