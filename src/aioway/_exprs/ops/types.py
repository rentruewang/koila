# Copyright (c) AIoWay Authors - All Rights Reserved

"The signature param list type."

import dataclasses as dcls
import functools
from collections.abc import Sequence
from typing import Self

from lark import Lark, Transformer

from aioway import _logging

from . import _common

__all__ = ["TypeList"]

LOGGER = _logging.get_logger(__name__)


class TypeList:
    """
    The parameter list in signature. Here we only care about types.
    This is useful to support type checking on ``Tensor`` related operations.
    """

    def __init__(self, *types: type) -> None:
        self._types: tuple[type, ...] = types
        "The types of the list of params."

        for t in types:
            if not isinstance(t, type):
                raise TypeError(f"{types=} expect all types, but some are not types.")

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

    def __getitem__(self, idx: int) -> type:
        return self._types[idx]

    def __iter__(self):
        return iter(self._types)

    def check_types(self, types: Sequence[type], /) -> bool:
        "Validate the given types."

        # Should have the same length.
        if len(self) != len(types):
            return False

        # Check if each item is instance of types.
        for t, i in zip(self, types):
            if not issubclass(i, t):
                return False

        return True

    def check_values(self, items: Sequence, /) -> bool:
        "Validate the given values."

        return self.check_types([type(elem) for elem in items])

    @classmethod
    def parse(cls, text: str, /, **types: type) -> Self:
        return _common.parse_and_transform_later(
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
    return Lark(_PARAM_LIST_GRAMMAR, start="param_list")


@_common.lark_transformer_dcls
class ParamListTransformer(Transformer):
    _mapping: dict[str, type] = dcls.field(default_factory=dict)

    def params(self, *names: str):
        types = (self._mapping[name] for name in names)
        return TypeList(*types)
