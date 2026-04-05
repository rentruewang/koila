# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing
from collections import abc as cabc

from aioway._tracking.logging import get_logger

from .getters import GetItemSymbol, SelectSymbol
from .ufuncs import (
    AddColSymbol,
    EqColSymbol,
    ExpColSymbol,
    FloorDivColSymbol,
    GeColSymbol,
    GtColSymbol,
    InvColSymbol,
    LeColSymbol,
    LtColSymbol,
    MultColSymbol,
    NeColSymbol,
    NegColSymbol,
    SubColSymbol,
    TrueDivColSymbol,
)

__all__ = ["Symbol", "ColSymbol", "TableSymbol"]

LOGGER = get_logger(__name__)


class Symbol(abc.ABC):
    """
    An (extended) projection operator that can be reprsented as an expression.
    """

    @abc.abstractmethod
    def __str__(self) -> str: ...

    @typing.final
    def _return_type(self):
        return str


class ColSymbol(Symbol, abc.ABC):
    @abc.abstractmethod
    def __str__(self) -> str: ...

    @typing.final
    def __invert__(self):
        return InvColSymbol(self)

    @typing.final
    def __neg__(self):
        return NegColSymbol(self)

    @typing.final
    def __add__(self, other: ColSymbol):
        return AddColSymbol(self, other)

    @typing.final
    def __sub__(self, other: ColSymbol):
        return SubColSymbol(self, other)

    @typing.final
    def __mul__(self, other: ColSymbol):
        return MultColSymbol(self, other)

    @typing.final
    def __truediv__(self, other: ColSymbol):
        return TrueDivColSymbol(self, other)

    @typing.final
    def __floordiv__(self, other: ColSymbol):
        return FloorDivColSymbol(self, other)

    @typing.final
    def __pow__(self, other: ColSymbol):
        return ExpColSymbol(self, other)

    @typing.final
    def __eq__(self, other: object):
        if isinstance(other, ColSymbol):
            return EqColSymbol(self, other)

        return NotImplemented

    @typing.final
    def __ne__(self, other: object):
        if isinstance(other, ColSymbol):
            return NeColSymbol(self, other)

        return NotImplemented

    @typing.final
    def __gt__(self, other: ColSymbol):
        return GtColSymbol(self, other)

    @typing.final
    def __ge__(self, other: ColSymbol):
        return GeColSymbol(self, other)

    @typing.final
    def __lt__(self, other: ColSymbol):
        return LtColSymbol(self, other)

    @typing.final
    def __le__(self, other: ColSymbol):
        return LeColSymbol(self, other)


class TableSymbol(Symbol, abc.ABC):

    @typing.overload
    def __getitem__(self, key: str, /) -> ColSymbol: ...

    @typing.overload
    def __getitem__(self, key: list[str], /) -> typing.Self: ...

    def __getitem__(self, key, /):
        match key:
            case str():
                return self.column(key)
            case list() if all(isinstance(i, str) for i in key):
                return self.select(*key)

        raise TypeError(
            "The default implemenetation of `Table.__getitem__` "
            f"does not know how to handle {key=}. "
            "It only supports `key` of type `str` and `list[str]`."
        )

    @abc.abstractmethod
    def __str__(self) -> str: ...

    @abc.abstractmethod
    def keys(self) -> cabc.KeysView[str]: ...

    def column(self, key: str) -> ColSymbol:
        return GetItemSymbol(table=self, column=key)

    def select(self, *keys: str) -> TableSymbol:
        return SelectSymbol(table=self, columns=keys)
