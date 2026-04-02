# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import operator
import typing

import torch

from aioway import _typing, tensors
from aioway._tracking import logging

from . import vectors

__all__ = ["VectorExpr"]

LOGGER = logging.get_logger(__name__)


type VectorExprRhs = VectorExpr | vectors.VectorRhs


@dcls.dataclass(frozen=True)
class VectorExpr:
    """
    The expression type for `vectors.Vector`.
    """

    tensor: _tensor_exprs.TensorExpr
    "The tensor expression type."

    attr: tensors.Attr
    "The attribute type (eagerly computed)."

    def __post_init__(self):
        if not isinstance(self.tensor, _tensor_exprs.TensorExpr):
            raise TypeError(
                f"{type(self.tensor)=} is not a `_tensor_exprs.TensorExpr`."
            )

        if not isinstance(self.attr, tensors.Attr):
            raise TypeError(f"{type(self.tensor)=} is not an `tensors.Attr`.")

    def __neg__(self) -> typing.Self:
        return self.__ufunc1(operator.neg)

    def __invert__(self) -> typing.Self:
        return self.__ufunc1(operator.invert)

    def __getitem__(self, key: int | slice | torch.Tensor | VectorExpr) -> typing.Self:
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

    def __add__(self, other: VectorExprRhs) -> typing.Self:
        return self.__ufunc2(other, operator.add)

    def __sub__(self, other: VectorExprRhs) -> typing.Self:
        return self.__ufunc2(other, operator.sub)

    def __mul__(self, other: VectorExprRhs) -> typing.Self:
        return self.__ufunc2(other, operator.mul)

    def __truediv__(self, other: VectorExprRhs) -> typing.Self:
        return self.__ufunc2(other, operator.truediv)

    def __floordiv__(self, other: VectorExprRhs) -> typing.Self:
        return self.__ufunc2(other, operator.floordiv)

    def __mod__(self, other: VectorExprRhs) -> typing.Self:
        return self.__ufunc2(other, operator.mod)

    def __pow__(self, other: VectorExprRhs) -> typing.Self:
        return self.__ufunc2(other, operator.pow)

    @typing.no_type_check
    def __eq__(self, other: VectorExprRhs) -> typing.Self:
        return self.__cmp(other, operator.eq)

    @typing.no_type_check
    def __ne__(self, other: VectorExprRhs) -> typing.Self:
        return self.__cmp(other, operator.ne)

    def __ge__(self, other: VectorExprRhs) -> typing.Self:
        return self.__cmp(other, operator.ge)

    def __gt__(self, other: VectorExprRhs) -> typing.Self:
        return self.__cmp(other, operator.gt)

    def __le__(self, other: VectorExprRhs) -> typing.Self:
        return self.__cmp(other, operator.le)

    def __lt__(self, other: VectorExprRhs) -> typing.Self:
        return self.__cmp(other, operator.lt)

    def _inputs(self):
        return (self.tensor,)

    def _return_type(self):
        return vectors.Vector

    def compute(self) -> vectors.Vector:
        return self._compute()

    def _compute(self) -> vectors.Vector:
        data = self.tensor.compute()

        return vectors.Vector(data=data, attr=self.attr)

    def __ufunc1(self, op: _typing.AnyUFunc1) -> typing.Self:
        LOGGER.debug("%s.%s called", self, op)
        result = type(self)(tensor=op(self.tensor), attr=op(self.attr.term).unpack())
        LOGGER.debug("%s.%s returned %s", self, op, result)
        return result

    def __ufunc2(self, other: VectorExprRhs, op: _typing.AnyUFunc2) -> typing.Self:
        LOGGER.debug("%s.%s(%s) called", self, op, other)
        result = type(self)(
            tensor=self._tensor_expr(other=other, op=op),
            attr=self._dtype_term(other=other, op=op).unpack(),
        )
        LOGGER.debug("%s.%s(%s) returned %s", self, op, other, result)
        return result

    def __cmp(self, other: VectorExprRhs, op: _typing.AnyUFunc2) -> typing.Self:
        return self.__ufunc2(other, op)

    def _tensor_expr(
        self, other: VectorExprRhs, op: _typing.AnyUFunc2
    ) -> _tensor_exprs.TensorExpr:
        match other:
            case VectorExpr(tensor=tensor):
                return op(self.tensor, tensor)
            case vectors.Vector():
                return op(self.tensor, _tensor_exprs.SourceTensorExpr(other.torch()))
            case int() | float() | bool():
                return op(self.tensor, other)
        raise TypeError(f"Do not know how to handle {type(other)=}.")

    def _dtype_term(
        self, other: VectorExprRhs, op: _typing.AnyUFunc2
    ) -> tensors.AttrTerm:
        match other:
            case VectorExpr(attr=attr) | vectors.Vector(attr=attr):
                return op(self.attr.term, attr.term)
            case int() | float() | bool():
                return op(self.attr.term, other)
        raise TypeError(f"Do not know how to handle {type(other)=}.")
