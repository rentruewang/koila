# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import logging
import typing
from graphlib import TopologicalSorter
from typing import Protocol

from aioway.ops import BatchGen, Thunk

from .caches import CacheExec
from .execs import Exec
from .ops import OpExec

__all__ = ["DagExec"]

LOGGER = logging.getLogger(__name__)


class DagExec(Exec, key="DAG"):
    r"""
    ``DagExec`` allows for overlapping dependency (DAG) in the dependency graph.

    Assuming we have the following dependency::

        A     # A produces some data.
       / \
      B   C   # B, C both depend on A.
       \ /
        D     # D depends on B and C.

    A has ``out_degs = 2``, B, C would have ``out_degs = 1``, D has ``out_degs = 0``.

    If B and C share the same A iterator, A would be out of sync with B and C,
    B's and C's data at index ``i`` would get ``A``'s data at ``2i`` and ``2i + 1``, respectively.

    Therefore, we need to duplicate A's iterator::

      A   A   # A duplicates.
      |   |
      B   C   # B, C both depend on A.
       \ /
        D     # D depends on B and C.


    Now there are 2 solutions::

        #. Just call ``iter(A)`` twice and make ``A`` compute 2 times.
           This duplicates computation but is safe. This would happen if we apply ``LazyExec``.

        #. Make A return the same thing twice.
           This is what ``DagExec`` uses. It inserts ``RepeatOp`` s.t. we have the following graph.

    Note:
        This only works if ``next(B)`` ``next(C)`` are called in succession,
        i.e. we don't call ``next(B)`` twice before calling ``next(C)``, and vice versa.
        Here we delegate the task to ``LazyExec`` s.t. it is guarenteed to only call on ``next(D)``,
        which ensures that before the next ``next(D)`` call, all the dependencies are finished.
    """

    def __init__(self, thunk: Thunk, /) -> None:
        self._thunk = thunk

    @typing.override
    def iterate(self) -> BatchGen:
        """
        Yields a generator, locally, from ``Op``'s definition.

        Returns:
            A stream of ``Block``s.

        Note:
            Always creates a new ``Generator`` upon being called, not cached.
        """

        yield from self.memoized_exec()

    def memoized_exec(self):
        thunk_deps = self._thunk.deps()
        dag_node_list = build_thunk_deps_dag(thunk_deps)

        # Transform the graph, to cache nodes that are ``iter``-ed more than once.
        return build_execs_with_cache(dag_node_list)

    @property
    @typing.override
    def thunk(self) -> Thunk:
        return self._thunk


@dcls.dataclass
class ThunkDagNode:
    thunk: Thunk

    out_nodes: list[Thunk] = dcls.field(default_factory=list)

    @property
    def in_nodes(self):
        return self.thunk.inputs

    @property
    def in_degs(self):
        return len(self.in_nodes)

    @property
    def out_degs(self) -> int:
        return len(self.out_nodes)


def build_execs_with_cache(node_list: list[ThunkDagNode]) -> Exec:
    """
    Build executors from topological order.

    If degree > 1, an ``CacheExec`` would be used instead,
    whose ``__iter__`` would build a ``TorchListFrame``,
    for the generator to iterate on.
    This means that they would cache the results in memory.
    """

    deps = build_dependency_graph(node_list)
    check_deps_is_topo_sorted(deps)
    rebuilt: list[Exec] = []

    # Rebuilding the entire tree, because ``Thunk``s are immutable.
    for idx in range(len(node_list)):
        node = node_list[idx]

        # Use ``LazyExec``'s recursive transformation.
        exe: Exec = OpExec(node.thunk.op, (rebuilt[n] for n in deps[idx]))

        # Only insert ``CacheExec`` when ``times > 1``, for efficiency.
        if node.out_degs > 1:
            exe = CacheExec(exe)

        rebuilt.append(exe)

    # [-1] is safe, because ``node_list`` cannot be empty,
    # as ``DagExec.thunk`` is specified.
    return rebuilt[-1]


def build_thunk_deps_dag(thunk_deps: set[Thunk], /) -> list[ThunkDagNode]:
    """
    Find all thunk dependencies, sort by topological order, build into a DAG.
    """

    node_list = [ThunkDagNode(thunk=t) for t in thunk_deps]
    deps = build_dependency_graph(node_list)

    # Sort the nodes.
    node_list = topo_sort(node_list, deps)

    # Check if the dependency graph of the new DAG is valid.
    deps = build_dependency_graph(node_list)
    check_deps_is_topo_sorted(deps)

    populate_out_nodes(node_list)
    return node_list


def check_deps_is_topo_sorted(deps: list[list[int]]):
    """
    This function would fail if the dependency graph is not topologically sorted,
    i.e. all indices in the deps is smaller than the index of current position.
    """

    for i, js in enumerate(deps):
        for j in js:
            assert i > j


def build_dependency_graph(node_list: list[ThunkDagNode]):
    """
    Build dependency graph from a node list.

    Returns:
        A list of list, each list contains a list of indices
        to the dependencies of the current node, in the current list.
    """

    deps: list[list[int]] = [[] for _ in range(len(node_list))]

    LOGGER.debug("Building the dependency dictionary.")

    def build_dependency_dict(node_idx: int, ipt_idx: int):
        deps[node_idx].append(ipt_idx)

    apply_for_node_input(node_list, build_dependency_dict)
    return deps


def topo_sort(node_list: list[ThunkDagNode], deps: list[list[int]]):
    """
    Topologically sort.
    """

    LOGGER.debug("Reorder by topological sort.")

    order = TopologicalSorter(graph=dict(enumerate(deps))).static_order()
    node_list = [node_list[i] for i in order]
    return node_list


def populate_out_nodes(node_list: list[ThunkDagNode]):
    LOGGER.debug("Populate the `out_nodes` for the DAG")

    def populate_output_for_node(node_idx: int, ipt_idx: int):
        node_list[ipt_idx].out_nodes.append(node_list[node_idx].thunk)

    apply_for_node_input(node_list, populate_output_for_node)


class _NodeInputPairFunc(Protocol):
    def __call__(self, node_idx: int, ipt_idx: int) -> None:
        """
        The function applies on the node index and its input's index, on a node list's index.
        """

        ...


def apply_for_node_input(node_list: list[ThunkDagNode], func: _NodeInputPairFunc):
    """
    Apply the ``function`` for each node and their inputs,
    which takes the node and its dependency's index in ``node_list``.

    Args:
        node_list: The node list of thunks.
        func:
            The function to apply each node and each of its input.
            The argument are the index of the node's position in the ``node_list``.
    """

    thunk_list = [node.thunk for node in node_list]

    # Build index for quick lookup.
    thunk_id_idx = {thunk: idx for idx, thunk in enumerate(thunk_list)}

    for thunk in thunk_list:
        for ipt in thunk.inputs:
            node_idx = thunk_id_idx[thunk]
            ipt_idx = thunk_id_idx[ipt]
            func(node_idx, ipt_idx)
