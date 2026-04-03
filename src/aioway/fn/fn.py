# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import enum
import typing

__all__ = ["Fn", "FnState"]


class FnState(enum.Enum):
    "The status of a `Later` object."

    PENDING = enum.auto()
    "The object is pending evaluation."

    DONE = enum.auto()
    "The object is evaluated."


class Fn[T](abc.ABC):
    """
    `Fn`s represent computation that shall be done later.

    Like Haskell's thunks, once evaluated,
    the value is stored in the `Fn` itself and never re-evaluated.
    The value shall be gone during GC.

    I was going to go for `Op` but it's used a lot in `torch`.
    """

    __match_args__: typing.ClassVar[tuple[str, ...]]

    @abc.abstractmethod
    def do(self) -> T:
        """
        Perform the computation that is represented by this `Fn`.

        Should recursively call the dependent `Fn.do` functions.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def deps(self) -> tuple[Fn[typing.Any], ...]:
        """
        The `Fn`s that must be evaluated before we can evaluate the current `Fn`.

        Calling `do` on the current `Fn` would recursively
        """

        raise NotImplementedError

    @property
    def is_leaf(self) -> bool:
        "Whether or not the thunk is dependent on other thunks. If not, it's a leaf."

        return not self.deps()
