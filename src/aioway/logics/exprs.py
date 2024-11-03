# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from collections.abc import Sequence
from typing import Generic, Protocol, TypeVar

from aioway.logics.trees import Node

_T = TypeVar("_T", covariant=True)
_E = TypeVar("_E", covariant=True)


class Expr(Node["Expr"], Protocol):
    """
    ``Expr`` represents an expression tree.

    There are 3 types of expressions, ``LeafExpr``, ``UnaryExpr``, and ``BinaryExpr``,
    correspond to different operators.

    Todo:
        Right now, the operators in unary and binary expressions are just strings.
        While this is very flexible, it does not provide any guarantees.
        I would like to make this an enum of some sort.
    """

    @property
    @abc.abstractmethod
    def sources(self) -> Sequence["Expr"]: ...

    @abc.abstractmethod
    def accept(self, visitor: "ExprVisitor[_T]", /) -> _T: ...


class ExprVisitor(Protocol[_T]):
    """
    The visitor for ``Expr``.

    Todo:
        Perhaps ``Visitor`` should have 2 type arguments.
    """

    def visit(self, expr: "Expr", /) -> _T:
        return expr.accept(self)

    @abc.abstractmethod
    def leaf(self, expr: "LeafExpr", /) -> _T: ...

    @abc.abstractmethod
    def unary(self, expr: "UnaryExpr", /) -> _T: ...

    @abc.abstractmethod
    def binary(self, expr: "BinaryExpr", /) -> _T: ...


@typing.final
@dcls.dataclass(frozen=True)
class LeafExpr(Expr, Generic[_E]):
    """
    Leaf node in an expression.
    """

    value: _E
    """
    The value of the dataclass.
    """

    @property
    def sources(self) -> tuple[()]:
        return ()

    def accept(self, visitor: ExprVisitor[_T]) -> _T:
        return visitor.leaf(self)


@typing.final
@dcls.dataclass(frozen=True)
class UnaryExpr(Expr):
    """
    Unary operator transforms its inputs.
    """

    op: str
    """
    The token for the operator itself.
    """

    operand: Expr
    """
    The operand of the expression.
    """

    @property
    def sources(self) -> tuple[Expr]:
        return (self.operand,)

    def accept(self, visitor: ExprVisitor[_T]) -> _T:
        return visitor.unary(self)


@typing.final
@dcls.dataclass(frozen=True)
class BinaryExpr(Expr):
    """
    Binary operators combine two inputs.
    """

    op: str
    """
    The token for the operator itself.
    """

    left: Expr
    """
    The lhs of the infix operator.
    """

    right: Expr
    """
    The rhs of the infix operator.
    """

    @property
    def sources(self) -> tuple[Expr, Expr]:
        return self.left, self.right

    def accept(self, visitor: ExprVisitor[_T]) -> _T:
        return visitor.binary(self)
