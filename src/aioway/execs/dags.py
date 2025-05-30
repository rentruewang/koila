# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
from collections.abc import Iterable, Sequence
from graphlib import TopologicalSorter
from typing import Self

from aioway.blocks import Block
from aioway.errors import AiowayError

from .execs import Exec


@dcls.dataclass(frozen=True)
class ExecDag:
    """
    ``ExecDag`` schedules the execution plan
    and ensures that the execution does not go out of order.
    """

    execs: Sequence[Exec]

    def __post_init__(self):
        if not self.execs:
            raise DagExecError("Execs must be a non-empty list.")

        if len(self.execs) != len(set(self.execs)):
            raise DagExecError("Execs must be a list of unique elements.")

        if not all(isinstance(exe, Exec) for exe in self.execs):
            raise DagExecError("All elements in execs must be of type `Exec`.")

    def __next__(self) -> list[Block]:
        result: list[Block] = []
        for exe in self.execs:
            block = next(exe)
            result.append(block)
        return result

    @classmethod
    def from_outputs(cls, outputs: list[Exec]) -> Self:
        """
        Create a DAG from the outputs of the given execs.
        """

        if not outputs:
            raise DagExecError("Outputs must be a non-empty list.")

        # First, visit all execs and their dependencies in the graph.
        all_seen: set[Exec] = set()
        for output in outputs:
            _visit_all_exec_deps(exec=output, visited=all_seen)

        # Then, sort them in topological order.
        return cls.topo_sort_execs(execs=all_seen)

    @classmethod
    def topo_sort_execs(cls, execs: Iterable[Exec]) -> Self:
        # Then, sort them in topological order.
        graph = {exe: [child for child in exe.children] for exe in execs}
        topo_sorter = TopologicalSorter(graph=graph)
        topo_sorter.prepare()
        sorted_exes = list(topo_sorter.static_order())
        return cls(execs=sorted_exes)


def _visit_all_exec_deps(exec: Exec, visited: set[Exec]) -> None:
    """
    Visit all execs and their dependencies in the DAG and apply the given function to each exec.
    """

    if exec in visited:
        return

    visited.add(exec)

    for child in exec.children:
        _visit_all_exec_deps(exec=child, visited=visited)


class DagExecError(AiowayError, ValueError): ...
