# Copyright (c) AIoWay Authors - All Rights Reserved

"The expression type."

import abc
import typing
from abc import ABC

__all__ = ["Expr"]


class Expr[T](ABC):
    """
    The expression type, containing signatures to connect to different types.

    ``Expression`` is the base class for all lazy computation of note in ``aioway``.
    """

    @typing.final
    def compute(self) -> T:
        """
        Evaluates the current expression eagerly.

        Raises:
            TypeError: If the output type does not match signature.

        Returns:
            T: The result yielded from the user defined ``_eager`` function.
        """

        result = self._compute()

        if not isinstance(result, self.return_type):
            raise TypeError(f"Output {type(result)=}, but {self.return_type=}.")

        return result

    @abc.abstractmethod
    def _compute(self) -> T:
        "Implementation of ``compute``."

        ...

    @property
    def return_type(self) -> type[T]:
        "The return type of ``compute``."

        return self._return_type()

    @abc.abstractmethod
    def _return_type(self) -> type[T]:
        "The implementation of ``return_type``."

        ...

    @property
    def inputs(self) -> tuple["Expr[T]", ...]:
        "The sub expressions. The length must match the signature's parameters."

        inputs = self._inputs()

        if not isinstance(inputs, tuple):
            raise TypeError("Inputs should be a `Sequence`.")

        if not all(isinstance(expr, Expr) for expr in inputs):
            expr_types = [type(expr) for expr in inputs]
            raise TypeError(
                f"Input expression types {expr_types} is not subclass of `Expr`."
            )

        return inputs

    @abc.abstractmethod
    def _inputs(self) -> tuple["Expr[T]", ...]:
        "The inputs of the"
        ...

    @property
    def argc(self) -> int:
        "``len(inputs)`` must be equal to ``argc``."

        return len(self.inputs)
