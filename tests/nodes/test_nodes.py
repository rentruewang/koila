# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls

import pytest

from aioway.nodes import Dag, Node


@dcls.dataclass(frozen=True)
class DataNode(Node):
    data: str

    _: dcls.KW_ONLY

    sources: tuple["DataNode", ...] = dcls.field(default_factory=tuple)

    # This method is defined s.t. tests can be more elegant.
    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.data == other
        return NotImplemented

    @property
    def children(self):
        return tuple(self.sources)


@pytest.fixture
def outputs():
    a = DataNode("a")
    b = DataNode("b", sources=(a,))
    c = DataNode("c", sources=(a,))
    d = DataNode("d", sources=(b, c))
    e = DataNode("e", sources=(d,))
    return d, e


@pytest.fixture
def dag(outputs):
    return Dag.from_outputs(outputs)


def test_dag_data_order(dag):
    assert dag[0] == "a"
    assert sorted([dag[1], dag[2]], key=lambda n: n.data) == ["b", "c"]
    assert dag[3] == "d"
    assert dag[4] == "e"
