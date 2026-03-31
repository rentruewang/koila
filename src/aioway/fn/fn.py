# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import functools
import typing
from abc import ABC
from collections.abc import Iterator
from enum import Enum
from enum import auto as Auto
from typing import Any, ClassVar

from . import deferral

__all__ = ["Fn", "FnState"]


class FnState(Enum):
    "The status of a `Later` object."

    PENDING = Auto()
    "The object is pending evaluation."

    EVALUATED = Auto()
    "The object is evaluated."


class Fn[P, T](ABC):
    """
    `Fn`s represent computation that shall be done later.

    Like Haskell's thunks, once evaluated,
    the value is stored in the `Fn` itself and never re-evaluated.
    The value shall be gone during GC.

    I was going to go for `Op` but it's used a lot in `torch`.
    """

    __match_args__: ClassVar[tuple[str, ...]]

    def __init__(self) -> None:
        super().__init__()

        self.__result: T | None = None

        self.__preview: P = self._preview()

    @typing.final
    def preview(self) -> P:
        return self.__preview

    @abc.abstractmethod
    def _preview(self) -> P:
        """
        The result, in fake mode.
        This is always available as fake results are computed during init.
        """

        raise NotImplementedError

    @typing.final
    def do(self) -> T:
        """
        Perform the computation.

        If the result is previously stored, use the stored result.
        """

        if self.__result is None:
            self.__result = self._do()

        return self.__result

    @abc.abstractmethod
    def _do(self) -> T:
        """
        Do the `torch` related operations.
        """

        raise NotImplementedError

    @functools.cached_property
    def deps(self):
        "The depedent `Exec`s."

        return tuple(self._deps())

    @property
    def is_leaf(self) -> bool:
        "Whether or not the thunk is dependent on other thunks. If not, it's a leaf."
        return not self.deps

    @abc.abstractmethod
    def _deps(self) -> Iterator[Fn[Any]]:
        """
        Return the depedent thunks.
        """

    @property
    def state(self) -> FnState:
        """
        If `do` has been called, return `EVALUATED`.
        Else return `PENDING`.
        """

        if self.__result is None:
            return FnState.PENDING
        else:
            return FnState.EVALUATED

    @staticmethod
    def defer(item):
        return deferral.defer(item)
