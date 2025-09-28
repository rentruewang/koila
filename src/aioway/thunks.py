# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import functools
import typing
from abc import ABC
from collections.abc import Callable, Iterator
from typing import ClassVar, Protocol

from rich.tree import Tree

from aioway._errors import AiowayError

if typing.TYPE_CHECKING:
    from aioway.ops import Op, Op0, Op1, Op2

__all__ = ["thunk", "Thunk", "Thunk0", "Thunk1", "Thunk2"]


@dcls.dataclass(frozen=True)
class Thunk(ABC):
    """
    The function pointer (``Op`` itself) and their arguments (other ``Thunk``s).

    ``Thunk``s have 3 classes, ``Thunk0``, ``Thunk1``, ``Thunk2``.
    This is to allow for easy destructuring of ``inputs`` and ``Op`` in the compiler.
    """

    ARGC: ClassVar[int]
    """
    The argument count of the current class.
    Must be the same as ``Op.ARGC``, and equal to ``len(list(inputs()))``.
    """

    __match_args__: ClassVar[tuple[str, ...]] = "op", "args"
    """
    This is to allow ``Thunk`` to be used in match destructuring.
    """

    op: "Op"
    """
    The function to be applied to.
    """

    class Visitor[T](Protocol):
        """
        The ``Visitor`` pattern / strategy for ``Op``.
        """

        def thunk_0(self, thunk: "Thunk0", /) -> T: ...

        def thunk_1(self, thunk: "Thunk1", /) -> T: ...

        def thunk_2(self, thunk: "Thunk2", /) -> T: ...

    def __post_init__(self) -> None:
        if self.op.ARGC != self.ARGC:
            raise ThunkArgError(
                f"Got {self.ARGC} input arguments, but expect {self.op.ARGC} ones."
            )

        inputs = list(self.inputs())

        if self.ARGC != len(inputs):
            raise ThunkArgError(f"{self.ARGC=}, but got {inputs}, with {len(inputs)=}.")

        for arg in inputs:
            if isinstance(arg, Thunk):
                continue

            # For those that are not ``Thunk``s.
            raise ThunkArgError(
                "Thunk only accepts `Thunk` objects as arguments, or none at all. "
                f"Got {[type(arg) for arg in self.inputs()]}"
            )

    def __rich__(self) -> Tree:
        tree = Tree(label=str(self.op))
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

    op: "Op0"
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

    op: "Op1"
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

    op: "Op2"
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


def thunk(op: "Op", /, *inputs: Thunk) -> Thunk:
    """
    Function to build ``Thunk`` instead of directly invoking the classes.
    """

    return op.accept(_thunk_factory())(*inputs)


@functools.cache
def _thunk_factory():
    from aioway.ops import Op

    class ThunkFactory(Op.Visitor[Callable[..., Thunk]]):
        """
        The factory visitor wrapping the current op,
        and returns a function to create the thunks,
        base on the argument count of each op.
        """

        @typing.override
        def op_0(self, op: "Op0"):
            return lambda: Thunk0(op)

        @typing.override
        def op_1(self, op: "Op1"):
            return lambda input: Thunk1(op=op, input=input)

        @typing.override
        def op_2(self, op: "Op2"):
            return lambda left, right: Thunk2(op=op, left=left, right=right)

    return ThunkFactory()


class ThunkArgError(AiowayError, TypeError): ...
