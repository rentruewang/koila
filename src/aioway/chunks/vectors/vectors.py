# Copyright (c) AIoWay Authors - All Rights Reserved

"`Vector` is a homogenious array of values."

import operator
import typing

import torch

from aioway import tensors
from aioway._tensor_exprs import SourceTensorExpr
from aioway._tracking import logging
from aioway._typing import AnyUFunc1, AnyUFunc2

if typing.TYPE_CHECKING:
    from . import exprs

__all__ = ["Vector"]

LOGGER = logging.get_logger(__name__)


type VectorRhs = Vector | torch.Tensor | int | float | bool


class Vector:
    """
    A `Vector` is a `torch.Tensor` plus its `tensors.Attr`.
    """

    __match_args__ = "data", "attr"

    def __init__(self, data: torch.Tensor, attr: tensors.Attr) -> None:
        self._attr = attr
        "The attribute that the `torch.Tensor` must satisfy."

        self._data = data
        "The underlying data."

    def __repr__(self):
        return f"Vector(data={self._data}, attr={self._attr})"

    @LOGGER.function("DEBUG")
    def __neg__(self) -> typing.Self:
        return self.__op1(operator.neg)

    @LOGGER.function("DEBUG")
    def __invert__(self) -> typing.Self:
        return self.__op1(operator.invert)

    @LOGGER.function("DEBUG")
    def __add__(self, other: exprs.VectorExprRhs) -> typing.Self:
        return self.__op2(other, operator.add)

    @LOGGER.function("DEBUG")
    def __sub__(self, other: exprs.VectorExprRhs) -> typing.Self:
        return self.__op2(other, operator.sub)

    @LOGGER.function("DEBUG")
    def __mul__(self, other: exprs.VectorExprRhs) -> typing.Self:
        return self.__op2(other, operator.mul)

    @LOGGER.function("DEBUG")
    def __truediv__(self, other: exprs.VectorExprRhs) -> typing.Self:
        return self.__op2(other, operator.truediv)

    @LOGGER.function("DEBUG")
    def __floordiv__(self, other: exprs.VectorExprRhs) -> typing.Self:
        return self.__op2(other, operator.floordiv)

    @LOGGER.function("DEBUG")
    def __mod__(self, other: exprs.VectorExprRhs) -> typing.Self:
        return self.__op2(other, operator.mod)

    @LOGGER.function("DEBUG")
    def __pow__(self, other: exprs.VectorExprRhs) -> typing.Self:
        return self.__op2(other, operator.pow)

    @typing.no_type_check
    @LOGGER.function("DEBUG")
    def __eq__(self, other: exprs.VectorExprRhs) -> typing.Self:
        return self.__op2(other, operator.eq)

    @typing.no_type_check
    @LOGGER.function("DEBUG")
    def __ne__(self, other: exprs.VectorExprRhs) -> typing.Self:
        return self.__op2(other, operator.ne)

    @typing.no_type_check
    @LOGGER.function("DEBUG")
    def __ge__(self, other: exprs.VectorExprRhs) -> typing.Self:
        return self.__op2(other, operator.ge)

    @typing.no_type_check
    @LOGGER.function("DEBUG")
    def __gt__(self, other: exprs.VectorExprRhs) -> typing.Self:
        return self.__op2(other, operator.gt)

    @typing.no_type_check
    @LOGGER.function("DEBUG")
    def __le__(self, other: exprs.VectorExprRhs) -> typing.Self:
        return self.__op2(other, operator.le)

    @typing.no_type_check
    @LOGGER.function("DEBUG")
    def __lt__(self, other: exprs.VectorExprRhs) -> typing.Self:
        return self.__op2(other, operator.lt)

    def __op1(self, unop: AnyUFunc1) -> typing.Self:
        return self.from_expr(unop(self.expr()))

    def __op2(self, other: exprs.VectorExprRhs, binop: AnyUFunc2) -> typing.Self:
        from . import exprs

        match other:
            case Vector():
                return self.from_expr(binop(self.expr(), other.expr()))
            case exprs.VectorExpr() | float() | int() | bool():
                return self.from_expr(binop(self.expr(), other))

        raise NotImplementedError

    def torch(self) -> torch.Tensor:
        "Get the `torch.Tensor` data that this `Vector` contains."
        return self._data

    def typeof(self) -> tensors.Attr:
        "Get the type information `tensors.Attr` of the `torch.Tensor` that this `Vector` represents."
        return self._attr

    def cpu(self):
        data = self._data.cpu()
        attr = tensors.Attr.parse(
            device="cpu",
            shape=self._attr.shape,
            dtype=self._attr.dtype,
        )
        return type(self)(data=data, attr=attr)

    def numpy(self):
        return self.torch().cpu().numpy()

    def tolist(self):
        return self.torch().tolist()

    def expr(self):
        from . import exprs

        return exprs.VectorExpr(
            tensor=SourceTensorExpr(self.torch()), attr=self.typeof()
        )

    @property
    def data(self):
        return self._data

    @property
    def attr(self):
        return self._attr

    @classmethod
    def from_expr(cls, expr: exprs.VectorExpr, /) -> typing.Self:
        vec = expr.compute()
        return cls(data=vec.data, attr=vec.attr)
