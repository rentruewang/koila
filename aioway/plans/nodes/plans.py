# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from abc import ABC
from collections.abc import Callable

from rich.tree import Tree

from .nodes import TreeNode

if typing.TYPE_CHECKING:
    pass

__all__ = ["PlanNode"]


class PlanNode[T: "PlanNode"](TreeNode[T], ABC):
    @abc.abstractmethod
    def __str__(self) -> str: ...

    @property
    @abc.abstractmethod
    def children(self) -> tuple[T, ...]: ...


# TODO: Move to reducers.
@dcls.dataclass(frozen=True)
class PlanExplainer:
    plan: PlanNode

    def __rich__(self, render: Callable[[PlanNode], str] = str) -> Tree:
        # TODO Use ``reprlib`` or ``pprint`` s.t. we do not rely on ``rich`` in explainer.

        tree = Tree(label=render(self.plan))

        for child in self.plan.children:
            sub_tree = type(self)(child).__rich__(render=render)
            tree.add(sub_tree)

        return tree
