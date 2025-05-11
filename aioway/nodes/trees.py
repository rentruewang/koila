# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import typing
from abc import ABC

from .displays import PlanDisplay

__all__ = ["TreeNode", "NullaryNode", "UnaryNode", "BinaryNode"]


class TreeNode[T: "TreeNode"](ABC):
    def __str__(self) -> str:
        display = PlanDisplay.str()
        return display(self)

    def __rich__(self):
        display = PlanDisplay.rich()
        return display(self)

    @property
    @abc.abstractmethod
    def children(self) -> tuple[T, ...]: ...


class NullaryNode[T: "TreeNode"](TreeNode[T], ABC):
    """
    A node that does not have any children.
    """

    @property
    @typing.override
    def children(self) -> tuple[()]:
        return ()


class UnaryNode[T: "TreeNode"](TreeNode[T], ABC):
    """
    A node that has one child.
    """

    @property
    @abc.abstractmethod
    def _child(self) -> T:
        """
        The child of the node.
        """

        ...

    @property
    @typing.override
    def children(self) -> tuple[T]:
        return (self._child,)


class BinaryNode[T: "TreeNode"](TreeNode[T], ABC):
    """
    A node that has two children.
    """

    @property
    @abc.abstractmethod
    def _left(self) -> T:
        """
        The left child of the node.
        """

        ...

    @property
    @abc.abstractmethod
    def _right(self) -> T:
        """
        The right child of the node.
        """

        ...

    @property
    @typing.override
    def children(self) -> tuple[T, T]:
        return self._left, self._right
