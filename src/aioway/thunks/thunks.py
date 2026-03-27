# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import functools
import typing
from abc import ABC
from collections.abc import Callable, Iterator
from enum import Enum
from enum import auto as Auto
from typing import Any, ClassVar

from aioway import _common

__all__ = ["ThunkStatus"]


class ThunkStatus(Enum):
    "The status of a thunk."

    PENDING = Auto()
    "The thunk is pending evaluation."

    READY = Auto()
    "The thunk is evaluated."


class Thunk[T](ABC):
    """
    `Thunk`s represent computation that shall be done later.

    Like Haskell, once evaluated,
    the value is stored in the `Thunk` itself and never re-evaluated.
    The value shall be gone during GC.
    """

    __match_args__: ClassVar[tuple[str, ...]]

    def __init__(self) -> None:
        self.__status = ThunkStatus.PENDING

    def force(self) -> T:
        """
        Force a `Thunk` to be evaluated. Return the result.
        Once evaluated, this would not cause evaluation again.
        """

        result = self.__result
        self.__status = ThunkStatus.READY
        return result

    @abc.abstractmethod
    def _evaluate(self) -> T:
        """
        Do the computation. The result of this function would be stored into `__result`.
        """

        ...

    @functools.cached_property
    def __result(self) -> T:
        """
        Store the result onto `self`.
        """

        return self._evaluate()

    @property
    def inputs(self) -> Iterator[Thunk[object]]:
        """
        Return the input thunks.
        """

        for arg in self.__match_args__:
            if not isinstance(attr := getattr(self, arg), Thunk):
                raise TypeError(f"self.{arg!s}={attr} is not a `Thunk`!")

            yield attr

    @property
    def status(self) -> ThunkStatus:
        """
        The current status of the `Thunk`.
        """

        return self.__status


class DataThunk[T](Thunk[T]):
    "Represents some static data. Mainly used in storing primitives."

    def __init__(self, data: T):
        super().__init__()

        self._data = data
        _ = self.force()
        assert self.status == ThunkStatus.READY

    def __repr__(self):
        return f"{self.__class__.__name__}({self.data!r})"

    @property
    def data(self) -> T:
        return self._data

    @typing.override
    def _evaluate(self) -> T:
        return self.data


class ComputeThunk[**P, T](Thunk[T]):
    "Represents some computation."

    def __init__(self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> None:
        super().__init__()

        self._func = func
        self._args = args
        self._kwargs = kwargs

    def __repr__(self) -> str:
        return _common.format_function(self.func, *self.args, **self.kwargs)

    @typing.override
    def _evaluate(self) -> T:
        return self.func(*self.args, **self.kwargs)

    @property
    def func(self) -> Callable[P, T]:
        return self._func

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._kwargs
