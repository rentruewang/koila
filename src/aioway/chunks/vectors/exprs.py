# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import operator
import typing
from typing import Self

from torch import Tensor

from aioway import _logging
from aioway._exprs import Expr
from aioway._tensors import SourceTensorExpr, TensorExpr
from aioway._typing import AnyUFunc1, AnyUFunc2
from aioway.attrs import Attr, AttrTerm

from .vectors import Vector, VectorRhs

__all__ = ["VectorExpr"]

LOGGER = _logging.get_logger(__name__)


type VectorExprRhs = VectorExpr | VectorRhs


@dcls.dataclass(frozen=True)
class VectorExpr(Expr[Vector]):
    """
    The expression type for `Vector`.
    """

    tensor: TensorExpr
    "The tensor expression type."

    attr: Attr
    "The attribute type (eagerly computed)."

    def __post_init__(self):
        if not isinstance(self.tensor, TensorExpr):
            raise TypeError(f"{type(self.tensor)=} is not a `TensorExpr`.")

        if not isinstance(self.attr, Attr):
            raise TypeError(f"{type(self.tensor)=} is not an `Attr`.")

    def __neg__(self) -> Self:
        return self.__ufunc1(operator.neg)

    def __invert__(self) -> Self:
        return self.__ufunc1(operator.invert)

    def __getitem__(self, key: int | slice | Tensor | VectorExpr) -> Self:
        if isinstance(key, VectorExpr):
            return type(self)(
                attr=self.attr.term[key.attr].unpack(),
                tensor=self.tensor[key.tensor],
            )

        else:
            return type(self)(
                attr=self.attr.term[key].unpack(),
                tensor=self.tensor[key],
            )

    def __add__(self, other: VectorExprRhs) -> Self:
        return self.__ufunc2(other, operator.add)

    def __sub__(self, other: VectorExprRhs) -> Self:
        return self.__ufunc2(other, operator.sub)

    def __mul__(self, other: VectorExprRhs) -> Self:
        return self.__ufunc2(other, operator.mul)

    def __truediv__(self, other: VectorExprRhs) -> Self:
        return self.__ufunc2(other, operator.truediv)

    def __floordiv__(self, other: VectorExprRhs) -> Self:
        return self.__ufunc2(other, operator.floordiv)

    def __mod__(self, other: VectorExprRhs) -> Self:
        return self.__ufunc2(other, operator.mod)

    def __pow__(self, other: VectorExprRhs) -> Self:
        return self.__ufunc2(other, operator.pow)

    @typing.no_type_check
    def __eq__(self, other: VectorExprRhs) -> Self:
        return self.__cmp(other, operator.eq)

    @typing.no_type_check
    def __ne__(self, other: VectorExprRhs) -> Self:
        return self.__cmp(other, operator.ne)

    def __ge__(self, other: VectorExprRhs) -> Self:
        return self.__cmp(other, operator.ge)

    def __gt__(self, other: VectorExprRhs) -> Self:
        return self.__cmp(other, operator.gt)

    def __le__(self, other: VectorExprRhs) -> Self:
        return self.__cmp(other, operator.le)

    def __lt__(self, other: VectorExprRhs) -> Self:
        return self.__cmp(other, operator.lt)

    @typing.override
    def _inputs(self) -> tuple[Expr, ...]:
        return (self.tensor,)

    def _return_type(self):
        return Vector

    @typing.override
    def _compute(self) -> Vector:
        data = self.tensor.compute()

        return Vector(data=data, attr=self.attr)

    def __ufunc1(self, op: AnyUFunc1) -> Self:
        LOGGER.debug("%s.%s called", self, op)
        result = type(self)(tensor=op(self.tensor), attr=op(self.attr.term).unpack())
        LOGGER.debug("%s.%s returned %s", self, op, result)
        return result

    def __ufunc2(self, other: VectorExprRhs, op: AnyUFunc2) -> Self:
        LOGGER.debug("%s.%s(%s) called", self, op, other)
        result = type(self)(
            tensor=self._tensor_expr(other=other, op=op),
            attr=self._dtype_term(other=other, op=op).unpack(),
        )
        LOGGER.debug("%s.%s(%s) returned %s", self, op, other, result)
        return result

    def __cmp(self, other: VectorExprRhs, op: AnyUFunc2) -> Self:
        return self.__ufunc2(other, op)

    def _tensor_expr(self, other: VectorExprRhs, op: AnyUFunc2) -> TensorExpr:
        match other:
            case VectorExpr(tensor=tensor):
                return op(self.tensor, tensor)
            case Vector():
                return op(self.tensor, SourceTensorExpr(other.torch()))
            case int() | float() | bool():
                return op(self.tensor, other)
        raise TypeError(f"Do not know how to handle {type(other)=}.")

    def _dtype_term(self, other: VectorExprRhs, op: AnyUFunc2) -> AttrTerm:
        match other:
            case VectorExpr(attr=attr) | Vector(attr=attr):
                return op(self.attr.term, attr.term)
            case int() | float() | bool():
                return op(self.attr.term, other)
        raise TypeError(f"Do not know how to handle {type(other)=}.")
