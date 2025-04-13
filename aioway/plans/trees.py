# Copyright (c) RenChu Wang - All Rights Reserved

__all__ = ["TreeNode"]

from typing import Protocol


class TreeNode[T: "TreeNode"](Protocol):
    @property
    def children(self: T) -> tuple[T, ...]:
        """
        The children of the same type.
        """

        ...
