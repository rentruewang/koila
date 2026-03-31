# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing
from abc import ABC
from collections import abc as cabc

from aioway._tracking import logging

__all__ = ["Symbol", "ColSymbol", "TableSymbol"]

LOGGER = logging.get_logger(__name__)


class Symbol(ABC):
    """
    An (extended) projection operator that can be reprsented as an expression.
    """

    @abc.abstractmethod
    def __str__(self) -> str: ...

    @typing.final
    def _return_type(self):
        return str


class ColSymbol(Symbol, ABC):
    @abc.abstractmethod
    def __str__(self) -> str: ...

    @typing.final
    def __invert__(self):
        from . import ufuncs

        return ufuncs.InvColSymbol(self)

    @typing.final
    def __neg__(self):
        from . import ufuncs

        return ufuncs.NegColSymbol(self)

    @typing.final
    def __add__(self, other: ColSymbol):
        from . import ufuncs

        return ufuncs.AddColSymbol(self, other)

    @typing.final
    def __sub__(self, other: ColSymbol):
        from . import ufuncs

        return ufuncs.SubColSymbol(self, other)

    @typing.final
    def __mul__(self, other: ColSymbol):
        from . import ufuncs

        return ufuncs.MultColSymbol(self, other)

    @typing.final
    def __truediv__(self, other: ColSymbol):
        from . import ufuncs

        return ufuncs.TrueDivColSymbol(self, other)

    @typing.final
    def __floordiv__(self, other: ColSymbol):
        from . import ufuncs

        return ufuncs.FloorDivColSymbol(self, other)

    @typing.final
    def __pow__(self, other: ColSymbol):
        from . import ufuncs

        return ufuncs.ExpColSymbol(self, other)

    @typing.final
    def __eq__(self, other: object):
        if isinstance(other, ColSymbol):
            from . import ufuncs

            return ufuncs.EqColSymbol(self, other)

        return NotImplemented

    @typing.final
    def __ne__(self, other: object):
        if isinstance(other, ColSymbol):
            from . import ufuncs

            return ufuncs.NeColSymbol(self, other)

        return NotImplemented

    @typing.final
    def __gt__(self, other: ColSymbol):
        from . import ufuncs

        return ufuncs.GtColSymbol(self, other)

    @typing.final
    def __ge__(self, other: ColSymbol):
        from . import ufuncs

        return ufuncs.GeColSymbol(self, other)

    @typing.final
    def __lt__(self, other: ColSymbol):
        from . import ufuncs

        return ufuncs.LtColSymbol(self, other)

    @typing.final
    def __le__(self, other: ColSymbol):
        from . import ufuncs

        return ufuncs.LeColSymbol(self, other)


class TableSymbol(Symbol, ABC):

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
        from . import getters

        return getters.GetItemSymbol(table=self, column=key)

    def select(self, *keys: str) -> TableSymbol:
        from . import getters

        return getters.SelectSymbol(table=self, columns=keys)
