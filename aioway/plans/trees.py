# Copyright (c) RenChu Wang - All Rights Reserved

from typing import Protocol

__all__ = ["TreeNode"]


class TreeNode[T: "TreeNode"](Protocol):
    @property
    def children(self: T) -> tuple[T, ...]:
        """
        The children of the same type.
        """

        ...
