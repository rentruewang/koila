# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
from collections import defaultdict as DefaultDict
from collections.abc import Hashable, Iterator
from graphlib import TopologicalSorter


@dcls.dataclass(frozen=True)
class Dag[T: Hashable]:
    deps: dict[T, set[T]] = dcls.field(default_factory=lambda: DefaultDict(set))
    """
    The predecessors / dependency of each item.
    The dependency graph must not contain a cycle, checked with ``TopologicalSorter``.
    """

    def __post_init__(self) -> None:
        # Ensure that there are not any cycles.
        ts = TopologicalSorter(self.deps)
        ts.prepare()

    def __contains__(self, node: T) -> bool:
        """
        Check if the node is present in the graph.

        Args:
            node: The node to check for.

        Returns:
            A boolean value.
        """

        return node in self.deps

    def add(self, node: T) -> None:
        """
        Add a node to the grpah.
        It is ok if the node is already present.

        Args:
            node: The node to add.
        """

        # Ensuring that the predecessors are also initialized,
        # using ``DefaultDict``'s behavior.
        _ = self.deps[node]

    def remove(self, node: T, /) -> None:
        """
        Remove a node from the graph.

        Args:
            node:
                The node from the graph to remove.
                It must be present in the graph prior to deletion,
                or else a ``KeyError`` would be raised.
        """

        del self.deps[node]

    def add_dep(self, node: T, /, *parents: T) -> None:
        """
        Add dependency / dependencies to the graph.
        Args:
            node: The node to add parents to.
            *parents: The nodes that are a direct dependency of ``node``.
        """

        self.deps[node] |= set(parents)

    def parents(self, node: T, /) -> Iterator[T]:
        """
        Get the parents of a node.
        """

        yield from self.deps[node]

    def children(self, node: T, /) -> Iterator[T]:
        """
        Get the children of a node.
        """

        for key, val in self.deps.items():
            if node in val:
                yield key

    def topo_sort(self) -> Iterator[T]:
        """
        Return the topological order of a graph.
        """

        ts = TopologicalSorter(self.deps)
        yield from ts.static_order()
