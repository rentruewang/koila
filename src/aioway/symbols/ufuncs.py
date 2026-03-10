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

    @property
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
        return f"{self.left!s} {self.token} {self.right!s}"

    @typing.override
    def _inputs(self):
        return self.left, self.right

    @property
    @abc.abstractmethod
    def token(self) -> str:
        "The token representing the current operator."

        ...


class NotColExpr(UFuncSymColExpr1, key="+"):

    @property
    @typing.override
    def token(self) -> str:
        return "~"


class NegColExpr(UFuncSymColExpr1, key="-"):
    @property
    @typing.override
    def token(self) -> str:
        return "-"


class AddColExpr(UFuncSymColExpr2, key="add"):
    @property
    @typing.override
    def token(self) -> str:
        return "+"


class SubColExpr(UFuncSymColExpr2, key="sub"):
    @property
    @typing.override
    def token(self) -> str:
        return "-"


class MultColExpr(UFuncSymColExpr2, key="mul"):
    @property
    @typing.override
    def token(self) -> str:
        return "*"


class TrueDivColExpr(UFuncSymColExpr2, key="truediv"):
    @property
    @typing.override
    def token(self) -> str:
        return "/"


class FloorDivColExpr(UFuncSymColExpr2, key="floordiv"):
    @property
    @typing.override
    def token(self) -> str:
        return "//"


class ExpColExpr(UFuncSymColExpr2, key="pow"):
    @property
    @typing.override
    def token(self) -> str:
        return "**"


class EqColExpr(UFuncSymColExpr2, key="eq"):
    @property
    @typing.override
    def token(self) -> str:
        return "=="


class NeColExpr(UFuncSymColExpr2, key="ne"):
    @property
    @typing.override
    def token(self) -> str:
        return "!="


class GeColExpr(UFuncSymColExpr2, key="ge"):
    @property
    @typing.override
    def token(self) -> str:
        return ">="


class LeColExpr(UFuncSymColExpr2, key="le"):
    @property
    @typing.override
    def token(self) -> str:
        return "<="


class GtColExpr(UFuncSymColExpr2, key="gt"):
    @property
    @typing.override
    def token(self) -> str:
        return ">"


class LtColExpr(UFuncSymColExpr2, key="lt"):
    @property
    @typing.override
    def token(self) -> str:
        return "<"
