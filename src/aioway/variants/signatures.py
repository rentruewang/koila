# Copyright (c) AIoWay Authors - All Rights Reserved

"The signature types."

import dataclasses as dcls
import functools
import logging
import typing
from collections.abc import Callable
from typing import Any

import lark
from lark import Lark, Transformer

__all__ = ["ParamList", "Signature"]

LOGGER = logging.getLogger(__name__)


class ParamList:
    "The parameter list in signature. Here we only care about types."

    def __init__(self, *types: type) -> None:
        self._types: tuple[type, ...] = types
        "The types of the list of params."

        for t in types:
            if not isinstance(t, type):
                raise TypeError(f"{types=} expect all types, but some are not types.")

    def __eq__(self, other: object):
        if isinstance(other, ParamList):
            return self._types == other._types

        if isinstance(other, list | tuple):
            return self._types == tuple(other)

        return NotImplemented

    def __repr__(self) -> str:
        return f"{type(self).__qualname__}{self!s}"

    def __str__(self) -> str:
        types_str = ", ".join(t.__qualname__ for t in self._types)
        return f"({types_str})"

    def __hash__(self):
        return hash(self._types)

    def __len__(self) -> int:
        return len(self._types)

    def __getitem__(self, idx: int):
        return self._types[idx]

    def __iter__(self):
        yield from self._types

    def check(self, *items: Any) -> bool:
        # Should have the same length.
        if len(self) != len(items):
            return False

        # Check if each item is instance of types.
        for t, i in zip(self, items):
            if not isinstance(i, t):
                return False

        return True

    @classmethod
    def parse(cls, text: str, /, **types: type):
        return _parse_and_transform_later(
            parser=_param_list_lark_parser,
            transformer=_ParamListTransformer,
            text=text,
        )(**types)


@dcls.dataclass(frozen=True)
class Signature:
    "The signature type."

    params: ParamList
    "The parameters of the signature."

    result: type
    "The return type of the signature."

    def __str__(self):
        return f"{self.params!s} -> {self.result.__qualname__}"

    @classmethod
    def parse(cls, text: str, /, **types: type):
        return _parse_and_transform_later(
            parser=_signature_lark_parser,
            transformer=_SignatureTransformer,
            text=text,
        )(**types)


_PARAM_LIST_GRAMMAR = r"""
?param_list: "(" params ")"
params: VAR_NAME ("," VAR_NAME)*
VAR_NAME: /[a-zA-Z_]\w*/

%import common.WS
%ignore WS
"""

_SIGNATURE_LIST_GRAMMAR = r"""
signature: param_list "->" VAR_NAME
""" + _PARAM_LIST_GRAMMAR


@typing.dataclass_transform()
def _inline_transform(cls):
    cls = dcls.dataclass(frozen=True)(cls)
    cls = lark.v_args(inline=True)(cls)
    return cls


@_inline_transform
class _ParamListTransformer(Transformer):
    _mapping: dict[str, type] = dcls.field(default_factory=dict)

    def params(self, *names: str):
        types = (self._mapping[name] for name in names)
        return ParamList(*types)


@_inline_transform
class _SignatureTransformer(_ParamListTransformer):
    def signature(self, param_list: ParamList, result: str):
        return Signature(params=param_list, result=self._mapping[result])


@functools.cache
def _param_list_lark_parser():
    return Lark(_PARAM_LIST_GRAMMAR, start="param_list")


@functools.cache
def _signature_lark_parser():
    return Lark(_SIGNATURE_LIST_GRAMMAR, start="signature")


def _parse_and_transform_later(
    parser: Callable[[], Lark],
    transformer: Callable[[dict[str, type]], Transformer],
    text: str,
):
    def later(**types: type):
        LOGGER.debug("Parsing %s", text)
        parsed = parser().parse(text)
        LOGGER.debug("Parsed: %s", parsed)
        result = transformer(types).transform(parsed)
        LOGGER.debug("Result: %s", result)
        return result

    return later
