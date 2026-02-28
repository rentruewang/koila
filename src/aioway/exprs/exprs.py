# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import logging
import typing
from collections.abc import Iterator, Sequence
from typing import ClassVar, Literal, Protocol

from aioway.tables import Table

__all__ = ["Expr", "ColumnExpr", "TableExpr"]

LOGGER = logging.getLogger(__name__)


class Expr(Protocol):
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


class ColumnExpr(Expr, Protocol):
    @abc.abstractmethod
    def subs[C](self, **tables: Table[C]) -> C: ...

    @typing.override
    def __repr__(self) -> str:
        return f"col({self!s})"


class TableExpr(Expr, Protocol):
    @abc.abstractmethod
    def subs[C](self, **tables: Table[C]) -> Table[C]: ...

    @typing.override
    def __repr__(self) -> str:
        return f"sel({self!s})"
