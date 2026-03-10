# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing
from abc import ABC

from aioway._exprs import OpSign

from . import _common
from .exprs import ColumnSymbolExpr

__all__ = [
    "UFuncSymColExpr1",
    "NegColExpr",
    "NotColExpr",
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
class UFuncSymColExpr1(ColumnSymbolExpr, ABC):
    source: ColumnSymbolExpr

    def __init_subclass__(cls, key: str) -> None:
        OpSign.ufunc1(ColumnSymbolExpr).register_keys(key)(cls)

    @typing.override
    def _compute(self) -> str:
        return f"{self.token} {self.source!s}"

    @typing.override
    def _inputs(self):
        return (self.source,)

    @abc.abstractmethod
    def token(self) -> str:
        "The token representing the current operator."

        ...


@_common.symbol_dataclass
class UFuncSymColExpr2(ColumnSymbolExpr, ABC):
    left: ColumnSymbolExpr
    right: ColumnSymbolExpr

    def __init_subclass__(cls, key: str) -> None:
        OpSign.ufunc2(ColumnSymbolExpr).register_keys(key)(cls)

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
class NotColExpr(UFuncSymColExpr1, key="+"):

    @typing.override
    def token(self) -> str:
        return "~"


@_common.symbol_dataclass
class NegColExpr(UFuncSymColExpr1, key="-"):

    @typing.override
    def token(self) -> str:
        return "-"


@_common.symbol_dataclass
class AddColExpr(UFuncSymColExpr2, key="add"):

    @typing.override
    def token(self) -> str:
        return "+"


@_common.symbol_dataclass
class SubColExpr(UFuncSymColExpr2, key="sub"):

    @typing.override
    def token(self) -> str:
        return "-"


@_common.symbol_dataclass
class MultColExpr(UFuncSymColExpr2, key="mul"):

    @typing.override
    def token(self) -> str:
        return "*"


@_common.symbol_dataclass
class TrueDivColExpr(UFuncSymColExpr2, key="truediv"):

    @typing.override
    def token(self) -> str:
        return "/"


@_common.symbol_dataclass
class FloorDivColExpr(UFuncSymColExpr2, key="floordiv"):

    @typing.override
    def token(self) -> str:
        return "//"


@_common.symbol_dataclass
class ExpColExpr(UFuncSymColExpr2, key="pow"):

    @typing.override
    def token(self) -> str:
        return "**"


@_common.symbol_dataclass
class EqColExpr(UFuncSymColExpr2, key="eq"):

    @typing.override
    def token(self) -> str:
        return "=="


@_common.symbol_dataclass
class NeColExpr(UFuncSymColExpr2, key="ne"):

    @typing.override
    def token(self) -> str:
        return "!="


@_common.symbol_dataclass
class GeColExpr(UFuncSymColExpr2, key="ge"):

    @typing.override
    def token(self) -> str:
        return ">="


@_common.symbol_dataclass
class LeColExpr(UFuncSymColExpr2, key="le"):

    @typing.override
    def token(self) -> str:
        return "<="


@_common.symbol_dataclass
class GtColExpr(UFuncSymColExpr2, key="gt"):

    @typing.override
    def token(self) -> str:
        return ">"


@_common.symbol_dataclass
class LtColExpr(UFuncSymColExpr2, key="lt"):

    @typing.override
    def token(self) -> str:
        return "<"
