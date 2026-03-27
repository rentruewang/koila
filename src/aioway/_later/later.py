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

__all__ = ["LaterStatus", "Later", "LaterData", "LaterCompute"]


class LaterStatus(Enum):
    "The status of a `Later` object."

    PENDING = Auto()
    "The object is pending evaluation."

    EVALUATED = Auto()
    "The object is evaluated."


class Later[T](ABC):
    """
    `Later`s represent computation that shall be done later.

    Like Haskell, once evaluated,
    the value is stored in the `Thunk` itself and never re-evaluated.
    The value shall be gone during GC.
    """

    __match_args__: ClassVar[tuple[str, ...]]

    def __init__(self) -> None:
        self.__state = LaterStatus.PENDING

    def do(self) -> T:
        """
        Force a `Defer` to be evaluated. Return the result.
        Once evaluated, this would not cause evaluation again.
        """

        result = self.__result
        self.__state = LaterStatus.EVALUATED
        return result

    @abc.abstractmethod
    def _do(self) -> T:
        """
        Do the computation. The result of this function would be stored into `__result`.
        """

        ...

    @functools.cached_property
    def __result(self) -> T:
        """
        Store the result onto `self`.
        """

        return self._do()

    @functools.cached_property
    def deps(self):
        "The depedent `Exec`s."

        return tuple(self._deps())

    @property
    def is_leaf(self) -> bool:
        "Whether or not the thunk is dependent on other thunks. If not, it's a leaf."
        return not self.deps

    @abc.abstractmethod
    def _deps(self) -> Iterator[Later[Any]]:
        """
        Return the depedent thunks.
        """

    @property
    def state(self) -> LaterStatus:
        """
        The current status of the `Thunk`.
        """

        return self.__state


class LaterData[T](Later[T]):
    "Represents some static data. Mainly used in storing primitives."

    __match_args__ = ("data",)

    def __init__(self, data: T):
        super().__init__()

        self._data = data
        _ = self.do()
        assert self.state == LaterStatus.EVALUATED

    def __repr__(self):
        return f"{self.__class__.__name__}({self.data!r})"

    @property
    def data(self) -> T:
        return self._data

    @typing.override
    def _do(self) -> T:
        return self.data

    @typing.override
    def _deps(self) -> Iterator[Later[object]]:
        return
        yield


class LaterCompute[T](Later[T]):
    "Represents some computation that is deferred."

    __match_args__ = "func", "args", "kwargs"

    def __init__(
        self, func: Callable[..., T], *args: Later[Any], **kwargs: Later[Any]
    ) -> None:
        super().__init__()

        self._func = func

        for arg in self.args:
            if not isinstance(arg, Later):
                raise TypeError(f"{arg} is not a `Thunk`.")

        for key, value in self.kwargs.items():
            if not isinstance(value, Later):
                raise TypeError(f"{key}={value} is not a `Thunk`")

        self._args = args
        self._kwargs = kwargs

    def __repr__(self) -> str:
        return _common.format_function(self.func, *self.args, **self.kwargs)

    @typing.override
    def _do(self) -> T:
        args = [arg.do() for arg in self.args]
        kwargs = {key: val.do() for key, val in self.kwargs.items()}
        return self.func(*args, **kwargs)

    @typing.override
    def _deps(self) -> Iterator[Later[object]]:
        yield from self.args
        yield from self.kwargs.values()

    @property
    def func(self) -> Callable[..., T]:
        return self._func

    @property
    def args(self) -> tuple[Later[Any], ...]:
        return self._args

    @property
    def kwargs(self) -> dict[str, Later[Any]]:
        return self._kwargs
