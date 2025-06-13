# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing
from typing import Protocol, Self

from .displays import PlanDisplay

if typing.TYPE_CHECKING:
    from aioway.compilers import Compiler

__all__ = ["Node", "NullaryNode", "UnaryNode", "BinaryNode"]


class Node[T: "Node"](Protocol):
    def __str__(self) -> str:
        display = PlanDisplay.str()
        return display(self)

    def __rich__(self):
        display = PlanDisplay.rich()
        return display(self)

    def rewrite[O: Node](self, compiler: "Compiler[Self, O]") -> O:
        return compiler(self)

    @property
    @abc.abstractmethod
    def children(self) -> tuple[T, ...]: ...


class NullaryNode[T: "Node"](Node[T], Protocol):
    """
    A node that does not have any children.
    """

    @property
    @typing.override
    def children(self) -> tuple[()]:
        return ()


class UnaryNode[T: "Node"](Node[T], Protocol):
    """
    A node that has one child.
    """

    child: T
    """
    The child of the node.
    """

    @property
    @typing.override
    def children(self) -> tuple[T]:
        return (self.child,)


class BinaryNode[T: "Node"](Node[T], Protocol):
    """
    A node that has two children.
    """

    left: T
    """
    The left child of the node.
    """

    right: T
    """
    The right child of the node.
    """

    @property
    @typing.override
    def children(self) -> tuple[T, T]:
        return self.left, self.right
