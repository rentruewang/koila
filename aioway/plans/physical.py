# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
from abc import ABC
from collections.abc import Callable

from rich.tree import Tree

from .trees import TreeNode

__all__ = ["PhysicalPlan", "PlanExplainer"]


class PhysicalPlan(TreeNode["PhysicalPlan"], ABC):
    @abc.abstractmethod
    def __str__(self) -> str: ...

    @property
    @abc.abstractmethod
    def children(self) -> tuple["PhysicalPlan", ...]: ...


@dcls.dataclass(frozen=True)
class PlanExplainer:
    plan: PhysicalPlan

    def tree(self, render: Callable[[PhysicalPlan], str] = str) -> Tree:
        tree = Tree(label=render(self.plan))

        for child in self.plan.children:
            sub_tree = type(self)(child).tree(render=render)
            tree.add(sub_tree)

        return tree
