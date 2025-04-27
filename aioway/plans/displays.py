# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import typing
from collections.abc import Callable

from rich.tree import Tree

if typing.TYPE_CHECKING:
    from .plans import PlanNode


def indent_children(node: str, children: list[str]) -> str:
    """Indent the children nodes for better readability."""
    indented_children = "\n".join(f"  {child}" for child in children)
    return f"{node}\n{indented_children}"


# TODO Make the API prettier.
@dcls.dataclass(frozen=True)
class PlanDisplay:
    def __call__[T](
        self,
        plan: "PlanNode",
        /,
        render: Callable[["PlanNode"], T],
        reduce: Callable[[T, list[T]], T],
    ) -> T:
        self_node = render(plan)

        children_nodes: list[T] = []
        for child in plan.children:
            sub_tree = self(child, render=render, reduce=reduce)
            children_nodes.append(sub_tree)

        return reduce(self_node, children_nodes)

    def str(self, plan: "PlanNode") -> str:
        return self(plan, render=lambda p: type(p).__qualname__, reduce=indent_children)

    def rich(self, plan: "PlanNode") -> Tree:
        render = lambda node: Tree(label=str(node))

        def reduce_tree(node: Tree, children: list[Tree]) -> Tree:
            for child in children:
                node.add(child)
            return node

        return self(plan, render=render, reduce=reduce_tree)
