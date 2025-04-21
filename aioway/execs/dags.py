# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
from collections.abc import Sequence

from .execs import Exec

__all__ = ["ExecDag"]


@dcls.dataclass(frozen=True)
class ExecDag:
    """
    A directed acyclic graph (DAG) of ``Exec``s.
    """

    outputs: Sequence[Exec]

    def backtrace(self) -> list[Exec]:
        """
        Backtrace the DAG to find all the inputs of the outputs.
        """

        visited: dict[int, Exec] = {}

        for out in self.outputs:
            self._backtrace(exec=out, visited=visited)

        return list(visited.values())

    def _backtrace(self, exec: Exec, visited: dict[int, Exec]) -> None:
        """
        Backtrace the DAG to find all the inputs of the outputs.
        """

        if hash(exec) in visited:
            return

        visited[hash(exec)] = exec

        for input in exec.children:
            self._backtrace(input, visited)
