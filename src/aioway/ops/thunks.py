# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
from typing import Self

from rich.tree import Tree

from aioway._errors import AiowayError

from .ops import Op

__all__ = ["Thunk", "thunk"]


@dcls.dataclass(frozen=True)
class Thunk:
    """
    The function pointer (``Op`` itself) and their arguments (other ``Thunk``s).
    """

    # __match_args__: ClassVar[tuple[str, ...]]
    # """
    # This is to allow ``Thunk`` to be used in match destructuring.
    # """

    op: Op
    """
    The function to be applied to.
    """

    inputs: tuple[Self, ...]
    """
    The input arguments. Must have the same length as ``op.ARGC``.
    """

    def __post_init__(self) -> None:
        if self.op.ARGC != self.argc:
            raise ThunkArgError(
                f"Got {self.argc} input arguments, but expect {self.op.ARGC} ones."
            )

        for arg in self.inputs:
            if type(arg) != type(self):
                raise ThunkArgError(
                    "Thunk only accepts `Thunk` objects as arguments, or none at all. "
                    f"Got {[type(arg) for arg in self.inputs]}"
                )

    def __rich__(self) -> Tree:
        tree = Tree(label=str(self.op))
        for arg in self.inputs:
            tree.add(arg.__rich__())
        return tree

    def deps(self) -> set[Self]:
        """
        Recursively find all thunk dependencies.
        """

        visited: set[Self] = set()
        self._find_deps_rec(visited)
        return visited

    def _find_deps_rec(self, visited: set[Self]) -> None:
        if self in visited:
            return

        visited.add(self)
        for ipt in self.inputs:
            ipt._find_deps_rec(visited)

    @property
    def argc(self) -> int:
        return len(self.inputs)


def thunk(op: Op, /, *inputs: Thunk) -> Thunk:
    return Thunk(op=op, inputs=inputs)


def find_all_thunk_deps(thunk: Thunk) -> set[Thunk]:

    visited: set[Thunk] = set()
    _find_all_thunk_deps_rec(thunk, visited)
    return visited


def _find_all_thunk_deps_rec(thunk: Thunk, visited: set[Thunk]) -> None:
    if thunk in visited:
        return

    visited.add(thunk)
    for ipt in thunk.inputs:
        _find_all_thunk_deps_rec(ipt, visited=visited)


class ThunkArgError(AiowayError, TypeError): ...
