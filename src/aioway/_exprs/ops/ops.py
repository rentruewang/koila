# Copyright (c) AIoWay Authors - All Rights Reserved

"The operators (output purely dependent on input and state)."

import dataclasses as dcls
import logging
import typing
from collections.abc import Callable
from typing import Any

from ..exprs import Expr
from .signs import OpSign, TypeList

__all__ = ["Op"]

LOGGER = logging.getLogger(__name__)

__all__ = ["Op", "OpExpr"]


@dcls.dataclass(frozen=True)
class Op[T: Callable]:
    """
    The type that is returned by the registry, contains both the callable and the signature.
    The operator must be stateless.
    """

    name: str
    "The name of the operator"

    signature: OpSign
    "The signature of the funciton."

    function: T
    "The callable that is registered."

    def __post_init__(self) -> None:
        LOGGER.debug(f"Operator {self!s} created.")

        if not callable(self.function):
            raise ValueError("The function is not callable.")

    def __str__(self):
        return f"{self.name}<{self.signature}>"

    def __call__(self, *args: Any) -> Any:
        if not self.signature.param_types.check_values(*args):
            return NotImplemented

        # Calling the function. Doesn't expect any errors.
        answer = self.function(*args)

        if not isinstance(answer, self.signature.return_type):
            raise AssertionError(
                f"Expected {self.signature.return_type}, got {type(answer)=}."
            )

        return answer

    @property
    def __func__(self) -> T:
        return self.function


class OpExpr[T](Expr[T]):
    """
    The operator signature.
    """

    def __init__(self, op: Op, *inputs: OpExpr) -> None:
        self._op = op
        self.__inputs = inputs

    @typing.override
    def _compute(self) -> Any:
        input_data = [i.compute() for i in self.inputs]
        self.param_types.check_values(input_data)
        return self._op.function(*input_data)

    @typing.override
    def _return_type(self) -> type[T]:
        return self._op.signature.return_type

    @property
    def param_types(self) -> TypeList:
        return self._op.signature.param_types

    @typing.override
    def _inputs(self):
        return self.__inputs
