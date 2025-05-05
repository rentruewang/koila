# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import typing
from abc import ABC

from .displays import PlanDisplay
from .nodes import TreeNode

__all__ = ["PlanNode"]


class PlanNode[T: "PlanNode"](TreeNode[T], ABC):
    def __str__(self) -> str:
        display = PlanDisplay.str()
        return display(self)

    def __rich__(self):
        display = PlanDisplay.rich()
        return display(self)

    @property
    @abc.abstractmethod
    @typing.override
    def children(self) -> tuple[T, ...]: ...
