# Copyright (c) AIoWay Authors - All Rights Reserved

"The signature param list type."

import dataclasses as dcls
import functools
import logging
import types
import typing

import lark

from ._common import lark_transformer_dcls, parse_and_transform_later

__all__ = ["TypeList"]

LOGGER = logging.getLogger(__name__)


class TypeList:
    """
    The parameter list in signature. Here we only care about types.
    This is useful to support type checking on `Tensor` related operations.
    """

    def __init__(self, *types: type | types.GenericAlias) -> None:
        self._types = types
        "The types of the list of params."

        LOGGER.debug("Initialized type list %r", self)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TypeList):
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
        return iter(self._types)

    @classmethod
    def parse(cls, text: str, /, **types: type) -> typing.Self:
        return parse_and_transform_later(
            parser=_param_list_lark_parser,
            transformer=ParamListTransformer,
            text=text,
        )(**types)

    @staticmethod
    def parsing_grammar():
        return _PARAM_LIST_GRAMMAR


_PARAM_LIST_GRAMMAR = r"""
?param_list: "(" params ")"
params: VAR_NAME ("," VAR_NAME)*
VAR_NAME: /[a-zA-Z_]\w*/

%import common.WS
%ignore WS
"""


@functools.cache
def _param_list_lark_parser():
    return lark.Lark(_PARAM_LIST_GRAMMAR, start="param_list")


@lark_transformer_dcls
class ParamListTransformer(lark.Transformer):
    _mapping: dict[str, type] = dcls.field(default_factory=dict)

    def params(self, *names: str):
        types = (self._mapping[name] for name in names)
        return TypeList(*types)
