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

__all__ = ["FnState", "Fn", "DataFn", "ThunkFn"]


class FnState(Enum):
    "The status of a `Later` object."

    PENDING = Auto()
    "The object is pending evaluation."

    EVALUATED = Auto()
    "The object is evaluated."


class Fn[T](ABC):
    """
    `Fn`s represent computation that shall be done later.

    Like Haskell's thunks, once evaluated,
    the value is stored in the `Fn` itself and never re-evaluated.
    The value shall be gone during GC.
    """

    __match_args__: ClassVar[tuple[str, ...]]

    def __init__(self) -> None:
        self.__state = FnState.PENDING

    def result(self) -> T:
        """
        Force a `Defer` to be evaluated. Return the result.
        Once evaluated, this would not cause evaluation again.
        """

        result = self.__result
        self.__state = FnState.EVALUATED
        return result

    @abc.abstractmethod
    def do(self) -> T:
        """
        Do the computation. The result of this function would be stored into `__result`.
        """

        ...

    @functools.cached_property
    def __result(self) -> T:
        """
        Store the result onto `self`.
        """

        return self.do()

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
        The current status of the `Thunk`.
        """

        return self.__state


class DataFn[T](Fn[T]):
    "Represents some static data. Mainly used in storing primitives."

    __match_args__ = ("data",)

    def __init__(self, data: T):
        super().__init__()

        self._data = data
        _ = self.result()
        assert self.state == FnState.EVALUATED

    def __repr__(self):
        return f"{self.__class__.__name__}({self.data!r})"

    @property
    def data(self) -> T:
        return self._data

    @typing.override
    def do(self) -> T:
        return self.data

    @typing.override
    def _deps(self) -> Iterator[Fn[object]]:
        return
        yield


class ThunkFn[T](Fn[T]):
    "Represents some computation that is deferred."

    __match_args__ = "func", "args", "kwargs"

    def __init__(
        self, func: Callable[..., T], *args: Fn[Any], **kwargs: Fn[Any]
    ) -> None:
        super().__init__()

        self._func = func

        for arg in self.args:
            if not isinstance(arg, Fn):
                raise TypeError(f"{arg} is not a `Thunk`.")

        for key, value in self.kwargs.items():
            if not isinstance(value, Fn):
                raise TypeError(f"{key}={value} is not a `Thunk`")

        self._args = args
        self._kwargs = kwargs

    def __repr__(self) -> str:
        return _common.format_function(self.func, *self.args, **self.kwargs)

    @typing.override
    def do(self) -> T:
        args = [arg.result() for arg in self.args]
        kwargs = {key: val.result() for key, val in self.kwargs.items()}
        return self.func(*args, **kwargs)

    @typing.override
    def _deps(self) -> Iterator[Fn[object]]:
        yield from self.args
        yield from self.kwargs.values()

    @property
    def func(self) -> Callable[..., T]:
        return self._func

    @property
    def args(self) -> tuple[Fn[Any], ...]:
        return self._args

    @property
    def kwargs(self) -> dict[str, Fn[Any]]:
        return self._kwargs
