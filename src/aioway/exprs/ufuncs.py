# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from abc import ABC
from collections.abc import Iterator

from aioway.exprs.exprs import Expr

from .exprs import ColumnExpr

__all__ = [
    "BinaryUFuncColExpr",
    "AddColExpr",
    "SubColExpr",
    "MultColExpr",
    "TrueDivColExpr",
    "FloorDivColExpr",
    "ExpColExpr",
    "EqColExpr",
    "NeColExpr",
    "GeColExpr",
    "LeColExpr",
    "GtColExpr",
    "LtColExpr",
]


@dcls.dataclass(frozen=True, eq=False)
class BinaryUFuncColExpr(ColumnExpr, ABC):
    NUM_ARGS = 2

    left: ColumnExpr
    "The LHS of the current operator."

    right: ColumnExpr
    "The RHS of the current operator."

    @typing.override
    def __str__(self) -> str:
        return f"{self.left!s} {self.token} {self.right!s}"

    @property
    @abc.abstractmethod
    def token(self) -> str:
        "The token representing the current operator."

        ...

    @typing.final
    def _children(self) -> Iterator[Expr]:
        yield self.left
        yield self.right


class AddColExpr(BinaryUFuncColExpr):
    @property
    @typing.override
    def token(self) -> str:
        return "+"


class SubColExpr(BinaryUFuncColExpr):
    @property
    @typing.override
    def token(self) -> str:
        return "-"


class MultColExpr(BinaryUFuncColExpr):
    @property
    @typing.override
    def token(self) -> str:
        return "*"


class TrueDivColExpr(BinaryUFuncColExpr):
    @property
    @typing.override
    def token(self) -> str:
        return "/"


class FloorDivColExpr(BinaryUFuncColExpr):
    @property
    @typing.override
    def token(self) -> str:
        return "//"


class ExpColExpr(BinaryUFuncColExpr):
    @property
    @typing.override
    def token(self) -> str:
        return "**"


class EqColExpr(BinaryUFuncColExpr):
    @property
    @typing.override
    def token(self) -> str:
        return "=="


class NeColExpr(BinaryUFuncColExpr):
    @property
    @typing.override
    def token(self) -> str:
        return "!="


class GeColExpr(BinaryUFuncColExpr):
    @property
    @typing.override
    def token(self) -> str:
        return ">="


class LeColExpr(BinaryUFuncColExpr):
    @property
    @typing.override
    def token(self) -> str:
        return "<="


class GtColExpr(BinaryUFuncColExpr):
    @property
    @typing.override
    def token(self) -> str:
        return ">"


class LtColExpr(BinaryUFuncColExpr):
    @property
    @typing.override
    def token(self) -> str:
        return "<"
