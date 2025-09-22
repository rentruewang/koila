# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import logging
import typing
from collections.abc import Iterator, Sequence
from types import EllipsisType
from typing import Self

import lark
from lark import Lark, LarkError, Transformer

from aioway._errors import AiowayError

__all__ = ["EinsumSignature", "EinsumParser"]
LOGGER = logging.getLogger(__name__)


EINSUM_GRAMMAR = r"""
?start: expr

expr: [ params ] "->" [ params ]

params: [ param ] ( "," [ param ] )*
?param: argument | ellipsis

argument: /[a-zA-Z0-9:]+/
ellipsis: "..."

%import common.WS
%ignore WS
"""


@typing.final
@dcls.dataclass(eq=False, frozen=True, repr=False)
class EinsumSignature:
    """
    The parsed outcome of the ``EinsumParser``,
    with each parameter and result split into list of strings.
    """

    params: tuple[str | EllipsisType, ...]
    """
    The input parameters of the einsum, parsed.
    """

    results: tuple[str | EllipsisType, ...]
    """
    The output parameters of the einsum, parsed.
    """

    def __post_init__(self) -> None:
        self.__check_if_args_are_strings()
        self.__check_ellipsis_only_once()

    def __check_if_args_are_strings(self) -> None:
        for val in self.in_out():
            if isinstance(val, str):
                continue

            raise EinsumSignInitError(
                f"Value '{val}' must be of type string. Got {type(val)=}"
            )

    def __check_ellipsis_only_once(self) -> None:
        if self.params.count(...) > 1 or self.results.count(...) > 1:
            raise EinsumSignInitError("'...' can only appear once in input or output")

    def __repr__(self) -> str:
        joined_params = ",".join(map(str, self.params))
        joined_results = ",".join(map(str, self.results))
        return f"{joined_params}->{joined_results}"

    def __eq__(self, other: object) -> bool:
        # We only need to compare the string representation
        # because we only wanted to check those.
        return str(self) == str(other)

    def in_out(self) -> Iterator[str]:
        """
        The input and output values of the ``Einsum``.
        """

        yield from map(str, self.params)
        yield from map(str, self.results)

    @property
    def num_inputs(self) -> int:
        """
        The number of input parameters.
        """

        return len(self.params)

    @property
    def num_outputs(self) -> int:
        """
        The number of output parameters.
        """

        return len(self.results)

    @classmethod
    def init(
        cls, params: Sequence[str] | str | None, results: Sequence[str] | str | None
    ) -> Self:
        """
        Initialize with both the parameters and the results.
        """

        LOGGER.debug(
            "Creating `Einsum` for params: %s and results: %s", params, results
        )

        params = _convert_to_sequence(params)
        results = _convert_to_sequence(results)
        return cls(params=params, results=results)


def _convert_to_sequence(seq: Sequence[str] | str | None) -> tuple[str, ...]:
    # Filter out both `None` and empty sequence,
    # as those won't be valid empty input / empty output.
    seq = seq or ""

    # Convert strings to a singleton.
    if isinstance(seq, str):
        return (seq,)

    if isinstance(seq, Sequence) and all(isinstance(s, str) for s in seq):
        return tuple(seq)

    raise EinsumSignInitError(f"Unknown type: {type(seq)=}")


@lark.v_args(inline=True)
class EinsumTransformer(Transformer):
    """
    The transformer corresponding to lark grammar.
    """

    expr = EinsumSignature.init
    argument = lambda self, text: str(text) if text else ""
    params = lambda self, *args: [arg if arg else "" for arg in args]
    ellipsis = lambda self, _: ...


@dcls.dataclass(frozen=True)
class EinsumParser:
    """
    The ``lark``-powered einsum parser.
    """

    parser: Lark
    """
    The parser of the class
    """

    def __call__(self, expr: str) -> EinsumSignature:
        """
        Parse the given expression.

        If the grammar parser errors out, raise ``EinsumError (ValueError)``.

        Args:
            expr: The expression as a string.
            grammar:
                The grammar string. Users do not need to modify.
                Defaults to EINSUM_GRAMMAR.

        Raises:
            EinsumError:
                If the parsing of ``lark`` throws an error,
                or if constructing ``Einsum`` throws an error,
                which happens when grammar is ok but semantic is not.

        Returns:
            An ``Einsum`` instance.
        """

        try:
            return self._parse(expr)
        except LarkError as le:
            raise EinsumParserError(f"Parsing expression failed: {expr}.") from le

    def _parse(self, text: str) -> EinsumSignature:
        LOGGER.debug("Parsing expression: %s", text)

        tree = self.parser.parse(text)

        LOGGER.debug("Expression %s parsed into %s. Transforming.", text, tree)
        result = EinsumTransformer().transform(tree)

        LOGGER.debug("Result parsed: %s", result)
        return result

    @classmethod
    def init(cls, *, grammar: str = EINSUM_GRAMMAR) -> Self:
        lark = Lark(grammar=grammar)
        return cls(parser=lark)


class EinsumParserError(AiowayError, ValueError): ...


class EinsumSignInitError(AiowayError, TypeError): ...
