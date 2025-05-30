# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing
from collections.abc import Callable

from rich.tree import Tree

if typing.TYPE_CHECKING:
    from .trees import TreeNode


@dcls.dataclass(frozen=True)
class PlanDisplay[T]:
    """
    ``PlanDisplay`` is a utility class for displaying the plan tree, recursively.
    It does so by recursively reducing the tree in a post-order traversal.
    """

    render: Callable[["TreeNode"], T]
    """
    The renderer function to be applied to each node in the tree.
    """

    reduce: Callable[[T, list[T]], T]
    """
    Combination of the node and its children.
    """

    def __call__(self, plan: "TreeNode", /) -> T:
        self_node = self.render(plan)

        children_nodes: list[T] = []
        for child in plan.children:
            sub_tree = self(child)
            children_nodes.append(sub_tree)
        return self.reduce(self_node, children_nodes)

    @staticmethod
    def str() -> "PlanDisplay[str]":
        """
        A simple string representation of the plan tree,
        this would print the tree as an indented list of nodes.
        """

        render = lambda node: type(node).__qualname__

        def indent_children(node: str, children: list[str]) -> str:
            """Indent the children nodes for better readability."""
            indented_children = "\n".join(f"  {child}" for child in children)
            return f"{node}\n{indented_children}"

        return PlanDisplay(render=render, reduce=indent_children)

    @staticmethod
    def rich() -> "PlanDisplay[Tree]":
        """
        A rich representation of the plan tree,
        this would use the `rich` library to create a tree structure.
        """

        render = lambda node: Tree(label=type(node).__qualname__)

        def reduce_tree(node: Tree, children: list[Tree]) -> Tree:
            for child in children:
                node.add(child)
            return node

        return PlanDisplay(render=render, reduce=reduce_tree)
