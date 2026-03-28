# Copyright (c) AIoWay Authors - All Rights Reserved

"`Vector` is a homogenious array of values."

import operator
import typing
from typing import Self

from torch import Tensor

from aioway._previews import Attr, _validation
from aioway._tensor_exprs import SourceTensorExpr
from aioway._tracking import logging
from aioway._typing import AnyUFunc1, AnyUFunc2

if typing.TYPE_CHECKING:
    from .exprs import VectorExpr, VectorExprRhs

__all__ = ["Vector"]

LOGGER = logging.get_logger(__name__)


type VectorRhs = Vector | Tensor | int | float | bool


class Vector:
    """
    A `Vector` is a `Tensor` plus its `Attr`.
    """

    __match_args__ = "data", "attr"

    def __init__(self, data: Tensor, attr: Attr) -> None:
        # Validate the attribute.
        _validation.validate_attr(attr=attr, tensor=data)

        self._attr = attr
        "The attribute that the `Tensor` must satisfy."

        self._data = data
        "The underlying data."

    def __repr__(self):
        return f"Vector(data={self._data}, attr={self._attr})"

    @LOGGER.function("DEBUG")
    def __neg__(self) -> Self:
        return self.__op1(operator.neg)

    @LOGGER.function("DEBUG")
    def __invert__(self) -> Self:
        return self.__op1(operator.invert)

    @LOGGER.function("DEBUG")
    def __add__(self, other: VectorExprRhs) -> Self:
        return self.__op2(other, operator.add)

    @LOGGER.function("DEBUG")
    def __sub__(self, other: VectorExprRhs) -> Self:
        return self.__op2(other, operator.sub)

    @LOGGER.function("DEBUG")
    def __mul__(self, other: VectorExprRhs) -> Self:
        return self.__op2(other, operator.mul)

    @LOGGER.function("DEBUG")
    def __truediv__(self, other: VectorExprRhs) -> Self:
        return self.__op2(other, operator.truediv)

    @LOGGER.function("DEBUG")
    def __floordiv__(self, other: VectorExprRhs) -> Self:
        return self.__op2(other, operator.floordiv)

    @LOGGER.function("DEBUG")
    def __mod__(self, other: VectorExprRhs) -> Self:
        return self.__op2(other, operator.mod)

    @LOGGER.function("DEBUG")
    def __pow__(self, other: VectorExprRhs) -> Self:
        return self.__op2(other, operator.pow)

    @typing.no_type_check
    @LOGGER.function("DEBUG")
    def __eq__(self, other: VectorExprRhs) -> Self:
        return self.__op2(other, operator.eq)

    @typing.no_type_check
    @LOGGER.function("DEBUG")
    def __ne__(self, other: VectorExprRhs) -> Self:
        return self.__op2(other, operator.ne)

    @typing.no_type_check
    @LOGGER.function("DEBUG")
    def __ge__(self, other: VectorExprRhs) -> Self:
        return self.__op2(other, operator.ge)

    @typing.no_type_check
    @LOGGER.function("DEBUG")
    def __gt__(self, other: VectorExprRhs) -> Self:
        return self.__op2(other, operator.gt)

    @typing.no_type_check
    @LOGGER.function("DEBUG")
    def __le__(self, other: VectorExprRhs) -> Self:
        return self.__op2(other, operator.le)

    @typing.no_type_check
    @LOGGER.function("DEBUG")
    def __lt__(self, other: VectorExprRhs) -> Self:
        return self.__op2(other, operator.lt)

    def __op1(self, unop: AnyUFunc1) -> Self:
        return self.from_expr(unop(self.expr()))

    def __op2(self, other: VectorExprRhs, binop: AnyUFunc2) -> Self:
        from .exprs import VectorExpr

        match other:
            case Vector():
                return self.from_expr(binop(self.expr(), other.expr()))
            case VectorExpr() | float() | int() | bool():
                return self.from_expr(binop(self.expr(), other))

        raise NotImplementedError

    def torch(self) -> Tensor:
        "Get the `Tensor` data that this `Vector` contains."
        return self._data

    def typeof(self) -> Attr:
        "Get the type information `Attr` of the `Tensor` that this `Vector` represents."
        return self._attr

    def cpu(self):
        data = self._data.cpu()
        attr = Attr.parse(
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
        from .exprs import VectorExpr

        return VectorExpr(tensor=SourceTensorExpr(self.torch()), attr=self.typeof())

    @property
    def data(self):
        return self._data

    @property
    def attr(self):
        return self._attr

    @classmethod
    def from_expr(cls, expr: VectorExpr, /) -> Self:
        vec = expr.compute()
        return cls(data=vec.data, attr=vec.attr)
