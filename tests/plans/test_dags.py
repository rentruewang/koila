# Copyright (c) RenChu Wang - All Rights Reserved

import json
from pathlib import Path

import pytest

from aioway.plans import Dag


def data_cases() -> dict[str, list[str]]:
    file = Path(__file__).parent / "cases.json"
    with file.open("r") as f:
        return json.load(f)


@pytest.fixture(params=data_cases(), scope="module")
def data(request) -> dict[str, list[str]]:
    return request.param


def test_dag(data):
    dag = Dag(data)
    ordered = tuple(dag.topo_sort())

    for key, val in data.items():
        for pred in val:
            assert ordered.index(key) > ordered.index(pred)
