# Copyright (c) AIoWay Authors - All Rights Reserved

"`Vector` is a homogenious array of values."

import operator
import typing

import torch

from aioway import _typing, meta, tensors
from aioway._tracking import logging

__all__ = ["Vector"]

LOGGER = logging.get_logger(__name__)


type VectorRhs = Vector | torch.Tensor | int | float | bool


class Vector:
    """
    A `Vector` is a `torch.Tensor` plus its `meta.Attr`.
    """

    __match_args__ = "data", "attr"

    def __init__(self, data: torch.Tensor) -> None:
        self._data = data
        "The underlying data."

    def __repr__(self):
        return f"Vector({self.attr})"

    @LOGGER.function("DEBUG")
    def __neg__(self) -> typing.Self:
        return self.__op1(operator.neg)

    @LOGGER.function("DEBUG")
    def __invert__(self) -> typing.Self:
        return self.__op1(operator.invert)

    @LOGGER.function("DEBUG")
    def __add__(self, other: typing.Any) -> typing.Self:
        return self.__op2(other, operator.add)

    @LOGGER.function("DEBUG")
    def __sub__(self, other: typing.Any) -> typing.Self:
        return self.__op2(other, operator.sub)

    @LOGGER.function("DEBUG")
    def __mul__(self, other: typing.Any) -> typing.Self:
        return self.__op2(other, operator.mul)

    @LOGGER.function("DEBUG")
    def __truediv__(self, other: typing.Any) -> typing.Self:
        return self.__op2(other, operator.truediv)

    @LOGGER.function("DEBUG")
    def __floordiv__(self, other: typing.Any) -> typing.Self:
        return self.__op2(other, operator.floordiv)

    @LOGGER.function("DEBUG")
    def __mod__(self, other: typing.Any) -> typing.Self:
        return self.__op2(other, operator.mod)

    @LOGGER.function("DEBUG")
    def __pow__(self, other: typing.Any) -> typing.Self:
        return self.__op2(other, operator.pow)

    @typing.no_type_check
    @LOGGER.function("DEBUG")
    def __eq__(self, other: typing.Any) -> typing.Self:
        return self.__op2(other, operator.eq)

    @typing.no_type_check
    @LOGGER.function("DEBUG")
    def __ne__(self, other: typing.Any) -> typing.Self:
        return self.__op2(other, operator.ne)

    @typing.no_type_check
    @LOGGER.function("DEBUG")
    def __ge__(self, other: typing.Any) -> typing.Self:
        return self.__op2(other, operator.ge)

    @typing.no_type_check
    @LOGGER.function("DEBUG")
    def __gt__(self, other: typing.Any) -> typing.Self:
        return self.__op2(other, operator.gt)

    @typing.no_type_check
    @LOGGER.function("DEBUG")
    def __le__(self, other: typing.Any) -> typing.Self:
        return self.__op2(other, operator.le)

    @typing.no_type_check
    @LOGGER.function("DEBUG")
    def __lt__(self, other: typing.Any) -> typing.Self:
        return self.__op2(other, operator.lt)

    def __op1(self, unop: typing.Any) -> typing.Self:
        return self.from_fn(unop(self.fn()))

    def __op2(self, other: typing.Any, binop: _typing.AnyUFunc2) -> typing.Self:
        pass

        match other:
            case Vector():
                return self.from_fn(binop(self.fn(), other.fn()))
            case tensors.TensorFn() | float() | int() | bool():
                return self.from_fn(binop(self.fn(), other))

        raise NotImplementedError

    def torch(self) -> torch.Tensor:
        "Get the `torch.Tensor` data that this `Vector` contains."
        return self._data

    def typeof(self) -> meta.Attr:
        "Get the type information `meta.Attr` of the `torch.Tensor` that this `Vector` represents."
        return self.attr

    def cpu(self) -> typing.Self:
        data = self._data.cpu()
        return type(self)(data=data)

    def numpy(self):
        return self.torch().cpu().numpy()

    def tolist(self):
        return self.torch().tolist()

    def fn(self):
        return tensors.TensorFn.from_tensor(self.data)

    @property
    def data(self):
        return self._data

    @property
    def attr(self):
        return meta.attr(self.data)

    @classmethod
    def from_fn(cls, fn: tensors.TensorFn, /) -> typing.Self:
        vec = fn.forward()
        return cls(data=vec.data)
