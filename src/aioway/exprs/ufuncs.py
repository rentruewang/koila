# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from abc import ABC
from collections.abc import Iterator

from aioway import variants
from aioway.variants import ParamList

from .exprs import ColumnExpr, Expr

__all__ = [
    "UnaryUFuncColExpr",
    "NegColExpr",
    "NotColExpr",
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
class UnaryUFuncColExpr(ColumnExpr, ABC):
    expr: ColumnExpr
    "The child of the current operator"

    @classmethod
    def __init_subclass__(cls, key: str) -> None:
        variants.register(ParamList(ColumnExpr), key)(cls)

    @typing.override
    def __str__(self) -> str:
        return f"{self.token} {self.expr!s}"

    @property
    @abc.abstractmethod
    def token(self) -> str:
        "The token representing the current operator."

        ...

    @typing.final
    def _children(self) -> Iterator[Expr]:
        yield self.expr


class NotColExpr(UnaryUFuncColExpr, key="not"):
    @property
    @typing.override
    def token(self) -> str:
        return "~"


class NegColExpr(UnaryUFuncColExpr, key="neg"):
    @property
    @typing.override
    def token(self) -> str:
        return "-"


@dcls.dataclass(frozen=True, eq=False)
class BinaryUFuncColExpr(ColumnExpr, ABC):
    NUM_ARGS = 2

    left: ColumnExpr
    "The LHS of the current operator."

    right: ColumnExpr
    "The RHS of the current operator."

    @classmethod
    def __init_subclass__(cls, key: str) -> None:
        variants.register(ParamList(ColumnExpr, ColumnExpr), key)(cls)

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


class AddColExpr(BinaryUFuncColExpr, key="add"):
    @property
    @typing.override
    def token(self) -> str:
        return "+"


class SubColExpr(BinaryUFuncColExpr, key="sub"):
    @property
    @typing.override
    def token(self) -> str:
        return "-"


class MultColExpr(BinaryUFuncColExpr, key="mul"):
    @property
    @typing.override
    def token(self) -> str:
        return "*"


class TrueDivColExpr(BinaryUFuncColExpr, key="truediv"):
    @property
    @typing.override
    def token(self) -> str:
        return "/"


class FloorDivColExpr(BinaryUFuncColExpr, key="floordiv"):
    @property
    @typing.override
    def token(self) -> str:
        return "//"


class ExpColExpr(BinaryUFuncColExpr, key="pow"):
    @property
    @typing.override
    def token(self) -> str:
        return "**"


class EqColExpr(BinaryUFuncColExpr, key="eq"):
    @property
    @typing.override
    def token(self) -> str:
        return "=="


class NeColExpr(BinaryUFuncColExpr, key="ne"):
    @property
    @typing.override
    def token(self) -> str:
        return "!="


class GeColExpr(BinaryUFuncColExpr, key="ge"):
    @property
    @typing.override
    def token(self) -> str:
        return ">="


class LeColExpr(BinaryUFuncColExpr, key="le"):
    @property
    @typing.override
    def token(self) -> str:
        return "<="


class GtColExpr(BinaryUFuncColExpr, key="gt"):
    @property
    @typing.override
    def token(self) -> str:
        return ">"


class LtColExpr(BinaryUFuncColExpr, key="lt"):
    @property
    @typing.override
    def token(self) -> str:
        return "<"
