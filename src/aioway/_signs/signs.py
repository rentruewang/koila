# Copyright (c) AIoWay Authors - All Rights Reserved

"The signature types."

import functools
import types
import typing

import lark

from aioway._tracking.logging import get_logger

from ._common import lark_transformer_dcls, parse_and_transform_later
from .tlists import ParamListTransformer, TypeList

__all__ = ["Signature"]

LOGGER = get_logger(__name__)


class Signature:
    """
    The signature type, used to describe the computation in a single expression node.

    Signature can be represented as (*param_types) -> return_type,
    where each param_types are the typed output of the previous expression.
    """

    __match_args__ = "param_types", "return_type"

    def __init__(self, *types: type | types.GenericAlias) -> None:
        *param_types, ret = types

        self._param_types = TypeList(*param_types)

        if not isinstance(ret, type):
            raise TypeError(
                f"The last argument to `Signature` must be a type. Got {ret}."
            )

        self._return_type = ret

    def __eq__(self, other: object):
        if isinstance(other, Signature):
            return (
                True
                and self._param_types == other._param_types
                and self._return_type == other._return_type
            )

        return NotImplemented

    def __repr__(self) -> str:
        return f"{self.param_types!s} -> {self.return_type.__qualname__}"

    def __hash__(self) -> int:
        return hash((self.param_types, self.return_type))

    @property
    def param_types(self) -> TypeList:
        "The parameters of the signature."

        return self._param_types

    @property
    def return_type(self) -> type:
        "The return type of the signature."

        return self._return_type

    @classmethod
    def parse(cls, text: str, /, **types: type) -> typing.Self:
        return parse_and_transform_later(
            parser=_signature_lark_parser,
            transformer=SignatureTransformer,
            text=text,
        )(**types)


_SIGNATURE_LIST_GRAMMAR = r"""
signature: param_list "->" VAR_NAME
""" + TypeList.parsing_grammar()


@lark_transformer_dcls
class SignatureTransformer(ParamListTransformer):
    def signature(self, param_list: TypeList, result: str):
        return Signature(*param_list, self._mapping[result])


@functools.cache
def _signature_lark_parser():
    return lark.Lark(_SIGNATURE_LIST_GRAMMAR, start="signature")
