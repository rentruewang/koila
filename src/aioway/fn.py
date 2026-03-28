# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import functools
import typing
from abc import ABC
from collections.abc import Iterator
from enum import Enum
from enum import auto as Auto
from typing import Any, ClassVar

from aioway import fake

__all__ = ["Fn", "FnState"]


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

    I was going to go for `Op` but it's used a lot in `torch`.
    """

    __match_args__: ClassVar[tuple[str, ...]]

    def __init__(self) -> None:
        super().__init__()

        self._real_result: T | None = None

        with fake.enable():
            self._fake_result: T = self.forward()

    @typing.final
    def do(self) -> T:
        """
        Perform the computation.

        If the result is previously stored, use the stored result.
        If we are in fake mode, return the `FakeTensor` version.

        Returns:
            A `FakeTensor` if in fake mode, a `Tensor` otherwise.
        """

        if fake.is_enabled():
            return self._fake_result

        if self._real_result is None:
            self._real_result = self.forward()

        return self._real_result

    @abc.abstractmethod
    def forward(self) -> T:
        """
        Do the `torch` related operations.
        This would yield `Fake*` in fake mode, but real tensor in real mode.
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

        if self._real_result is None:
            return FnState.PENDING
        else:
            return FnState.EVALUATED
