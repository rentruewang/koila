# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import abc
from typing import Protocol, TypeVar

from .nodes import Node

_T = TypeVar("_T", bound=Node, contravariant=True)
_E = TypeVar("_E", covariant=True)


class Walker(Protocol[_T, _E]):
    """
    ``Walker`` walks over a graph and convert it to something else.
    """

    @abc.abstractmethod
    def __call__(self, node: _T, /) -> _E:
        """
        Converts from the input (of a tree type) to an output.

        Args
            tree: The root node of the tree object to convert.

        Returns:
            A value (of which type is defined by a generic argument).
        """

        ...
