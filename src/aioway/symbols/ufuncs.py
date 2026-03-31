# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing

from . import _common, exprs

__all__ = [
    "UFuncSymColSymbol1",
    "NegColSymbol",
    "InvColSymbol",
    "UFuncSymColSymbol2",
    "AddColSymbol",
    "SubColSymbol",
    "MultColSymbol",
    "TrueDivColSymbol",
    "FloorDivColSymbol",
    "ExpColSymbol",
    "EqColSymbol",
    "NeColSymbol",
    "GeColSymbol",
    "LeColSymbol",
    "GtColSymbol",
    "LtColSymbol",
]


@_common.symbol_dataclass
class UFuncSymColSymbol1(exprs.ColSymbol, abc.ABC):
    source: exprs.ColSymbol

    @typing.override
    def __str__(self) -> str:
        return f"{self.token()}{self.source!s}"

    @abc.abstractmethod
    def token(self) -> str:
        "The token representing the current operator."

        raise NotImplementedError


@_common.symbol_dataclass
class UFuncSymColSymbol2(exprs.ColSymbol, abc.ABC):
    left: exprs.ColSymbol
    "The lhs of the expression. Must be `ColSymSymbol` because it corresponds to `self`."

    right: exprs.ColSymbol | int | float | bool
    "The rhs of the expression. Can be either `ColSymSymbol` or primitive types."

    @typing.override
    def __str__(self) -> str:
        return f"{self.left!s} {self.token()} {self.right!s}"

    @abc.abstractmethod
    def token(self) -> str:
        "The token representing the current operator."

        raise NotImplementedError


@_common.symbol_dataclass
class InvColSymbol(UFuncSymColSymbol1):

    @typing.override
    def token(self) -> str:
        return "~"


@_common.symbol_dataclass
class NegColSymbol(UFuncSymColSymbol1):

    @typing.override
    def token(self) -> str:
        return "-"


@_common.symbol_dataclass
class AddColSymbol(UFuncSymColSymbol2):

    @typing.override
    def token(self) -> str:
        return "+"


@_common.symbol_dataclass
class SubColSymbol(UFuncSymColSymbol2):

    @typing.override
    def token(self) -> str:
        return "-"


@_common.symbol_dataclass
class MultColSymbol(UFuncSymColSymbol2):

    @typing.override
    def token(self) -> str:
        return "*"


@_common.symbol_dataclass
class TrueDivColSymbol(UFuncSymColSymbol2):

    @typing.override
    def token(self) -> str:
        return "/"


@_common.symbol_dataclass
class FloorDivColSymbol(UFuncSymColSymbol2):

    @typing.override
    def token(self) -> str:
        return "//"


@_common.symbol_dataclass
class ExpColSymbol(UFuncSymColSymbol2):

    @typing.override
    def token(self) -> str:
        return "**"


@_common.symbol_dataclass
class EqColSymbol(UFuncSymColSymbol2):

    @typing.override
    def token(self) -> str:
        return "=="


@_common.symbol_dataclass
class NeColSymbol(UFuncSymColSymbol2):

    @typing.override
    def token(self) -> str:
        return "!="


@_common.symbol_dataclass
class GeColSymbol(UFuncSymColSymbol2):

    @typing.override
    def token(self) -> str:
        return ">="


@_common.symbol_dataclass
class LeColSymbol(UFuncSymColSymbol2):

    @typing.override
    def token(self) -> str:
        return "<="


@_common.symbol_dataclass
class GtColSymbol(UFuncSymColSymbol2):

    @typing.override
    def token(self) -> str:
        return ">"


@_common.symbol_dataclass
class LtColSymbol(UFuncSymColSymbol2):

    @typing.override
    def token(self) -> str:
        return "<"
