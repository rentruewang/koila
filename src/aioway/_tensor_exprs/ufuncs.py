# Copyright (c) AIoWay Authors - All Rights Reserved

import operator
from collections import abc as cabc

import torch

from . import _common
from .exprs import TensorExpr, TensorExprRhs

__all__ = ["TensorExpr1", "TensorExpr2"]


@_common.expr_dcls
class TensorExpr1(TensorExpr):
    __match_args__ = ("source",)

    name: str
    source: TensorExpr
    op: cabc.Callable[[torch.Tensor], torch.Tensor]

    def __repr__(self):
        return f"{self.name}{self.source!r}"

    def _inputs(self):
        return (self.source,)

    def _compute(self):
        source = self.source.compute()
        return self.op(source)

    @classmethod
    def invert(cls, source: TensorExpr):
        return cls("~", source=source, op=operator.invert)

    @classmethod
    def neg(cls, source: TensorExpr):
        return cls("-", source=source, op=operator.neg)


@_common.expr_dcls
class TensorExpr2(TensorExpr):
    __match_args__ = "left", "right"

    name: str
    left: TensorExpr
    right: TensorExpr | torch.Tensor | int | float | bool
    op: cabc.Callable[[torch.Tensor, torch.Tensor | int | float | bool], torch.Tensor]

    def __repr__(self):
        return f"{self.left!r} {self.name} {self.right!r}"

    def _inputs(self):
        return self.left, self.right

    def _compute(self):
        left = self.left.compute()

        if not isinstance(self.right, TensorExpr):
            right = self.right
        else:
            right = self.right.compute()

        return self.op(left, right)

    @classmethod
    def add(cls, left: TensorExpr, right: TensorExprRhs):
        return cls(name="+", left=left, right=right, op=operator.add)

    @classmethod
    def sub(cls, left: TensorExpr, right: TensorExprRhs):
        return cls(name="-", left=left, right=right, op=operator.sub)

    @classmethod
    def mul(cls, left: TensorExpr, right: TensorExprRhs):
        return cls(name="*", left=left, right=right, op=operator.mul)

    @classmethod
    def truediv(cls, left: TensorExpr, right: TensorExprRhs):
        return cls(name="/", left=left, right=right, op=operator.truediv)

    @classmethod
    def floordiv(cls, left: TensorExpr, right: TensorExprRhs):
        return cls(name="//", left=left, right=right, op=operator.floordiv)

    @classmethod
    def mod(cls, left: TensorExpr, right: TensorExprRhs):
        return cls(name="%", left=left, right=right, op=operator.mod)

    @classmethod
    def pow(cls, left: TensorExpr, right: TensorExprRhs):
        return cls(name="**", left=left, right=right, op=operator.pow)

    @classmethod
    def eq(cls, left: TensorExpr, right: TensorExprRhs):
        return cls(name="==", left=left, right=right, op=operator.eq)

    @classmethod
    def ne(cls, left: TensorExpr, right: TensorExprRhs):
        return cls(name="!=", left=left, right=right, op=operator.ne)

    @classmethod
    def ge(cls, left: TensorExpr, right: TensorExprRhs):
        return cls(name=">=", left=left, right=right, op=operator.ge)

    @classmethod
    def gt(cls, left: TensorExpr, right: TensorExprRhs):
        return cls(name=">", left=left, right=right, op=operator.gt)

    @classmethod
    def le(cls, left: TensorExpr, right: TensorExprRhs):
        return cls(name="<=", left=left, right=right, op=operator.le)

    @classmethod
    def lt(cls, left: TensorExpr, right: TensorExprRhs):
        return cls(name="<", left=left, right=right, op=operator.lt)
