# Copyright (c) RenChu Wang - All Rights Reserved

import abc
from typing import Protocol

from .nodes import Node

__all__ = ["Walker"]


class Walker[T: Node, E](Protocol):
    """
    ``Walker`` walks over a graph and convert it to something else.
    """

    @abc.abstractmethod
    def __call__(self, node: T, /) -> E:
        """
        Converts from the input (of a tree type) to an output.

        Args:
            tree: The root node of the tree object to convert.

        Returns:
            A value (of which type is defined by a generic argument).
        """

        ...
