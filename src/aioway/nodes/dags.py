# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import functools
from collections import defaultdict as DefaultDict
from collections.abc import Iterable, Mapping, Sequence
from graphlib import TopologicalSorter
from typing import Self

from aioway.errors import AiowayError

from .nodes import Node

__all__ = ["Dag"]


@dcls.dataclass
class DagAdj[T: Node]:
    """
    The adjacenct nodes to the current nodes in a DAG.
    """

    ins: list[T] = dcls.field(default_factory=list)
    outs: list[T] = dcls.field(default_factory=list)

    @property
    def in_degree(self) -> int:
        return len(self.ins)

    @property
    def out_degree(self) -> int:
        return len(self.outs)


@dcls.dataclass(frozen=True)
class Dag[T: Node]:
    """
    ``Dag`` orders the nodes with topological order.
    """

    nodes: Sequence[T]
    """
    The stored, ordered executors of the DAG.
    """

    def __post_init__(self):
        if not self.nodes:
            raise DagError("Execs must be a non-empty list.")

        if len(self.nodes) != len(set(self.nodes)):
            raise DagError("Execs must be a list of unique elements.")

        for exe in self.nodes:
            for child in exe.children:
                if self.index(exe) > self.index(child):
                    continue

                raise DagError(
                    "DAG's parent index must be greater than its children index."
                )

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(self, idx: int) -> T:
        return self.nodes[idx]

    def index(self, node: T, /) -> int:
        return self._rev_idx[node]

    @functools.cached_property
    def _rev_idx(self) -> Mapping[T, int]:
        return {node: idx for idx, node in enumerate(self.nodes)}

    @functools.cached_property
    def adj_list(self) -> Mapping[T, DagAdj]:
        adj: dict[T, DagAdj] = DefaultDict(DagAdj)
        for node in self.nodes:
            for child in node.children:
                adj[node].ins.append(child)
                adj[child].outs.append(node)
        return adj

    @functools.cached_property
    def inputs(self) -> tuple[T, ...]:
        return tuple(
            exe
            for exe, adj_for_node in self.adj_list.items()
            if adj_for_node.in_degree == 0
        )

    @functools.cached_property
    def outputs(self) -> tuple[T, ...]:
        return tuple(
            exe
            for exe, adj_for_node in self.adj_list.items()
            if adj_for_node.out_degree == 0
        )

    @classmethod
    def from_outputs(cls, outputs: Iterable[T], /) -> Self:
        """
        Create a DAG from the outputs of the given execs.
        """

        if not outputs:
            raise DagError("Outputs must be a non-empty list.")

        # First, visit all execs and their dependencies in the graph.
        all_seen: set[T] = set()
        for output in outputs:
            _visit_all_deps(exec=output, visited=all_seen)

        # Then, sort them in topological order.
        return cls.from_execs(all_seen)

    @classmethod
    def from_execs(cls, execs: Iterable[T], /) -> Self:
        """
        Create a DAG from a set of items.
        """

        # Then, sort them in topological order.
        graph = {exe: [child for child in exe.children] for exe in execs}
        topo_sorter = TopologicalSorter(graph=graph)
        sorted_exes = tuple(topo_sorter.static_order())
        return cls(nodes=sorted_exes)


def _visit_all_deps[T: Node](exec: T, visited: set[T]) -> None:
    """
    Visit all execs and their dependencies in the DAG and apply the given function to each exec.
    """

    if exec in visited:
        return

    visited.add(exec)

    for child in exec.children:
        _visit_all_deps(exec=child, visited=visited)


class DagError(AiowayError, ValueError): ...
