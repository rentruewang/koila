# Copyright (c) AIoWay Authors - All Rights Reserved

"The signature types."

import functools
import typing
from typing import Self

from lark import Lark

from aioway import _logging

from ..exprs import Expr
from . import _common
from .types import ParamListTransformer, TypeList

__all__ = ["OpSign", "OpSignExpr"]

LOGGER = _logging.get_logger(__name__)


class OpSign[T]:
    """
    The signature type, used to describe the computation in a single expression node.

    Signature can be represented as (*param_types) -> return_type,
    where each param_types are the typed output of the previous expression.
    """

    __match_args__ = "param_types", "return_type"

    def __init__(self, *types: type) -> None:
        *param_types, ret = types

        self._param_types = TypeList(*param_types)
        self._return_type = ret

    def __eq__(self, other: object):
        if isinstance(other, OpSign):
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
    def return_type(self) -> type[T]:
        "The return type of the signature."

        return self._return_type

    @classmethod
    def parse(cls, text: str, /, **types: type) -> Self:
        return _common.parse_and_transform_later(
            parser=_signature_lark_parser,
            transformer=SignatureTransformer,
            text=text,
        )(**types)

    @classmethod
    def ufunc0(cls, typ: type, /) -> Self:
        return cls(typ)

    @classmethod
    def ufunc1(cls, typ: type, /) -> Self:
        return cls(typ, typ)

    @classmethod
    def ufunc2(cls, typ: type, /) -> Self:
        return cls(typ, typ, typ)


@typing.final
class OpSignExpr(Expr[OpSign]):
    """
    The signature expression.

    The expression contains the full typing, when evaluated (``.compute()``),
    it would create signatures.
    """

    def __init__(self, return_type: type, *inputs: Self) -> None:
        """
        Args:
            return_type: The return type of the current signature.
            *inputs: The children of the current expression.
        """

        self.__inputs = inputs
        self.__return_type = return_type

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.__sub_strs} -> {self.return_type})"

    def __str__(self) -> str:
        return f"({self.__sub_strs}) -> {self.return_type.__name__}"

    @typing.override
    def _compute(self) -> OpSign:
        input_types = (i.return_type for i in self.inputs)
        return OpSign(*input_types, self.return_type)

    @typing.override
    def _return_type(self) -> type[OpSign]:
        return self.__return_type

    @typing.override
    def _inputs(self):
        return self.__inputs

    @property
    def __sub_strs(self):
        return ", ".join(map(str, self.inputs))

    @staticmethod
    def parsing_grammar():
        return _SIGNATURE_LIST_GRAMMAR


_SIGNATURE_LIST_GRAMMAR = r"""
signature: param_list "->" VAR_NAME
""" + TypeList.parsing_grammar()


@_common.lark_transformer_dcls
class SignatureTransformer(ParamListTransformer):
    def signature(self, param_list: TypeList, result: str):
        return OpSign(*param_list, self._mapping[result])


@functools.cache
def _signature_lark_parser():
    return Lark(_SIGNATURE_LIST_GRAMMAR, start="signature")
