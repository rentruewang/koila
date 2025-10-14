# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import functools
import typing
from abc import ABC
from collections.abc import Callable, Iterator
from typing import ClassVar, Protocol

from rich.tree import Tree

if typing.TYPE_CHECKING:
    from .plans import Plan, Plan0, Plan1, Plan2

__all__ = ["thunk", "Thunk", "Thunk0", "Thunk1", "Thunk2"]


@dcls.dataclass(frozen=True)
class Thunk(ABC):
    """
    The function pointer (``Plan`` itself) and their arguments (other ``Thunk``s).

    ``Thunk``s have 3 classes, ``Thunk0``, ``Thunk1``, ``Thunk2``.
    This is to allow for easy destructuring of ``inputs`` and ``Plan`` in the compiler.
    """

    ARGC: ClassVar[int]
    """
    The argument count of the current class.
    Must be the same as ``Plan.ARGC``, and equal to ``len(list(inputs()))``.
    """

    __match_args__: ClassVar[tuple[str, ...]] = "op", "args"
    """
    This is to allow ``Thunk`` to be used in match destructuring.
    """

    plan: "Plan"
    """
    The function to be applied to.
    """

    class Visitor[T](Protocol):
        """
        The ``Visitor`` pattern / strategy for ``Plan``.
        """

        def thunk_0(self, thunk: "Thunk0", /) -> T: ...

        def thunk_1(self, thunk: "Thunk1", /) -> T: ...

        def thunk_2(self, thunk: "Thunk2", /) -> T: ...

    def __post_init__(self) -> None:
        if self.plan.ARGC != self.ARGC:
            raise TypeError(
                f"Got {self.ARGC} input arguments, but expect {self.plan.ARGC} ones."
            )

        inputs = list(self.inputs())

        if self.ARGC != len(inputs):
            raise TypeError(f"{self.ARGC=}, but got {inputs}, with {len(inputs)=}.")

        for arg in inputs:
            if isinstance(arg, Thunk):
                continue

            # For those that are not ``Thunk``s.
            raise ValueError(
                "Thunk only accepts `Thunk` objects as arguments, or none at all. "
                f"Got {[type(arg) for arg in self.inputs()]}"
            )

    def __rich__(self) -> Tree:
        tree = Tree(label=str(self.plan))
        for arg in self.inputs():
            tree.add(arg.__rich__())
        return tree

    @abc.abstractmethod
    def inputs(self) -> Iterator["Thunk"]:
        """
        The input arguments. Must have the same length as ``op.ARGC``.
        """

    @property
    def args(self) -> list["Thunk"]:
        return list(self.inputs())

    def deps(self) -> set["Thunk"]:
        """
        Recursively find all thunk dependencies.
        """

        visited: set[Thunk] = set()
        self._find_deps_rec(visited)
        return visited

    def _find_deps_rec(self, visited: set["Thunk"]) -> None:
        if self in visited:
            return

        visited.add(self)
        for ipt in self.inputs():
            ipt._find_deps_rec(visited)


@typing.final
@dcls.dataclass(frozen=True)
class Thunk0(Thunk):
    ARGC = 0
    __match_args__ = ("op",)

    plan: "Plan0"
    """
    The function to be applied to.
    """

    @typing.override
    def inputs(self):
        return
        yield


@typing.final
@dcls.dataclass(frozen=True)
class Thunk1(Thunk):
    ARGC = 1
    __match_args__ = "op", "input"

    plan: "Plan1"
    """
    The function to be applied to.
    """

    input: Thunk
    "The only child of the current ``Thunk``."

    @typing.override
    def inputs(self):
        yield self.input


@typing.final
@dcls.dataclass(frozen=True)
class Thunk2(Thunk):
    ARGC = 2
    __match_args__ = "op", "left", "right"

    plan: "Plan2"
    """
    The function to be applied to.
    """

    left: Thunk
    "The left child of the current ``Thunk``."

    right: Thunk
    "The right child of the current ``Thunk``."

    @typing.override
    def inputs(self):
        yield self.left
        yield self.right


def thunk(op: "Plan", /, *inputs: Thunk) -> Thunk:
    """
    Function to build ``Thunk`` instead of directly invoking the classes.
    """

    return op.accept(_thunk_factory())(*inputs)


@functools.cache
def _thunk_factory():
    from .plans import Plan

    class ThunkFactory(Plan.Visitor[Callable[..., Thunk]]):
        """
        The factory visitor wrapping the current op,
        and returns a function to create the thunks,
        base on the argument count of each op.
        """

        @typing.override
        def plan_0(self, plan: "Plan0"):
            return lambda: Thunk0(plan)

        @typing.override
        def plan_1(self, plan: "Plan1"):
            return lambda input: Thunk1(plan=plan, input=input)

        @typing.override
        def plan_2(self, plan: "Plan2"):
            return lambda left, right: Thunk2(plan=plan, left=left, right=right)

    return ThunkFactory()
