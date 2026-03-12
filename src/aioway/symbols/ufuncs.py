# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing
from abc import ABC

from . import _common
from .exprs import ColSymExpr

__all__ = [
    "UFuncSymColExpr1",
    "NegColExpr",
    "InvColExpr",
    "UFuncSymColExpr2",
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


@_common.symbol_dataclass
class UFuncSymColExpr1(ColSymExpr, ABC):
    source: ColSymExpr

    @typing.override
    def _compute(self) -> str:
        return f"{self.token()}{self.source!s}"

    @typing.override
    def _inputs(self):
        return (self.source,)

    @abc.abstractmethod
    def token(self) -> str:
        "The token representing the current operator."

        ...


@_common.symbol_dataclass
class UFuncSymColExpr2(ColSymExpr, ABC):
    left: ColSymExpr
    "The lhs of the expression. Must be ``ColSymExpr`` because it corresponds to ``self``."

    right: ColSymExpr | int | float | bool
    "The rhs of the expression. Can be either ``ColSymExpr`` or primitive types."

    @typing.override
    def _compute(self) -> str:
        return f"{self.left!s} {self.token()} {self.right!s}"

    @typing.override
    def _inputs(self):
        return self.left, self.right

    @abc.abstractmethod
    def token(self) -> str:
        "The token representing the current operator."

        ...


@_common.symbol_dataclass
class InvColExpr(UFuncSymColExpr1):

    @typing.override
    def token(self) -> str:
        return "~"


@_common.symbol_dataclass
class NegColExpr(UFuncSymColExpr1):

    @typing.override
    def token(self) -> str:
        return "-"


@_common.symbol_dataclass
class AddColExpr(UFuncSymColExpr2):

    @typing.override
    def token(self) -> str:
        return "+"


@_common.symbol_dataclass
class SubColExpr(UFuncSymColExpr2):

    @typing.override
    def token(self) -> str:
        return "-"


@_common.symbol_dataclass
class MultColExpr(UFuncSymColExpr2):

    @typing.override
    def token(self) -> str:
        return "*"


@_common.symbol_dataclass
class TrueDivColExpr(UFuncSymColExpr2):

    @typing.override
    def token(self) -> str:
        return "/"


@_common.symbol_dataclass
class FloorDivColExpr(UFuncSymColExpr2):

    @typing.override
    def token(self) -> str:
        return "//"


@_common.symbol_dataclass
class ExpColExpr(UFuncSymColExpr2):

    @typing.override
    def token(self) -> str:
        return "**"


@_common.symbol_dataclass
class EqColExpr(UFuncSymColExpr2):

    @typing.override
    def token(self) -> str:
        return "=="


@_common.symbol_dataclass
class NeColExpr(UFuncSymColExpr2):

    @typing.override
    def token(self) -> str:
        return "!="


@_common.symbol_dataclass
class GeColExpr(UFuncSymColExpr2):

    @typing.override
    def token(self) -> str:
        return ">="


@_common.symbol_dataclass
class LeColExpr(UFuncSymColExpr2):

    @typing.override
    def token(self) -> str:
        return "<="


@_common.symbol_dataclass
class GtColExpr(UFuncSymColExpr2):

    @typing.override
    def token(self) -> str:
        return ">"


@_common.symbol_dataclass
class LtColExpr(UFuncSymColExpr2):

    @typing.override
    def token(self) -> str:
        return "<"
