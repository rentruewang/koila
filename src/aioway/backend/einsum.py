# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
import functools
import logging
import re
from collections.abc import Sequence
from re import Pattern
from typing import NamedTuple, Self

LOGGER = logging.getLogger(__name__)

_PARAMS = r"\w+"
_PARAM_LIST = r"[\w,]*"
_EINSUM = rf"({_PARAM_LIST})->({_PARAM_LIST})"


@functools.cache
def einsum_parser() -> Pattern:
    return re.compile(_EINSUM)


@functools.cache
def einsum_params() -> Pattern:
    return re.compile(_PARAMS)


class EinsumError(ValueError):
    """
    Errors related to einsum expressions.
    Thrown when an invalid einsum expression is given,
    or when the dimensions of the keys do not match the einsum expression.
    """


class EinsumExpr(NamedTuple):
    """
    Parsed einsum expression.
    Splits the einsum expression into inputs and outputs.
    """

    inputs: Sequence[str]
    "The input dimensions."

    outputs: Sequence[str]
    "The output dimensions."

    @property
    def dims(self) -> Sequence[str]:
        return [*self.inputs, *self.outputs]

    @classmethod
    def parse(cls, einsum: str) -> Self:
        """
        Parse the einsum expression and keys.

        Parameters:
            einsum: The einsum formatted string.

        Returns:
            An ``EinsumExpr`` instance.

        Raises:
            EinsumError: If the keys do not match the einsum expression.
        """

        LOGGER.debug("Parsing einsum expression, einsum=%s", einsum)

        parser = einsum_parser()
        params = einsum_params()

        if (result := parser.match(einsum)) is None:
            raise EinsumError(f"Invalid einsum expression: {einsum}.")

        if result.group() != einsum:
            raise EinsumError(f"`->` Not found in einsum expression: {einsum}")

        left_match, right_match = result.groups()

        left: list[str] = params.findall(left_match)
        right: list[str] = params.findall(right_match)

        return cls(left, right)


@dcls.dataclass(frozen=True)
class Einsum:
    """
    Einsum expression with bound variables.

    Todo:
        Add support for unbound variables.
    """

    einsum: EinsumExpr

    def __hash__(self) -> int:
        return hash(self.einsum)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Einsum):
            return NotImplemented

        return self.einsum == other.einsum

    def __call__(self, shapes: Sequence[tuple[int, ...]]) -> Sequence[tuple[int, ...]]:
        return self._shape(shapes)

    def _shape(self, shapes: Sequence[tuple[int, ...]]) -> Sequence[tuple[int, ...]]:
        """
        Compute the output shapes based on the einsum expression and the input shapes.
        """

        LOGGER.debug("Computing einsum shapes. einsum=%s, keys=%s", self.einsum, shapes)

        inputs = self.inputs

        if len(shapes) != len(inputs):
            raise EinsumError(
                "Number of arguments do not match the inputs of the einsum expression."
            )

        # The mapping from input dimension symbol to input dimension value.
        mapped: dict[str, int] = {}

        for dimensions, symbols in zip(shapes, inputs):
            if len(dimensions) != len(symbols):
                raise EinsumError(
                    f"Dimension of {dimensions} does not match the einsum expression."
                    " "
                    f"Expected {len(symbols)} for {symbols}, got {len(dimensions)}."
                )

            for key, dim in zip(symbols, dimensions):
                if key in mapped and mapped[key] != dim:
                    raise EinsumError(
                        f"Dimension of key {key} does not match the einsum expression."
                        " "
                        f"Expected {mapped[key]} for {key}, got {dim}."
                    )

                mapped[key] = dim

        return [tuple(mapped[d] for d in dims) for dims in self.outputs]

    @property
    def inputs(self) -> Sequence[str]:
        return self.einsum.inputs

    @property
    def outputs(self) -> Sequence[str]:
        return self.einsum.outputs

    @classmethod
    def parse(cls, einsum: str, /) -> Self:
        einsum_expr = EinsumExpr.parse(einsum)

        return cls(einsum_expr)
