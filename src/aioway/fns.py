# Copyright (c) AIoWay Authors - All Rights Reserved

"`Fn` is the base class for deferred operation."

import abc
import functools
from abc import ABC
from collections.abc import Iterator
from typing import Any, ClassVar

__all__ = ["Fn"]


class Fn[T](ABC):
    """
    `Fn`s represent computation that shall be done later.

    Like Haskell's thunks, once evaluated,
    the value is stored in the `Fn` itself and never re-evaluated.
    The value shall be gone during GC.

    I was going to go for `Op` but it's used a lot in `torch`.
    """

    __match_args__: ClassVar[tuple[str, ...]]

    @abc.abstractmethod
    def do(self) -> T:
        """
        Do the computation.
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
