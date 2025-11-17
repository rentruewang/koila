# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
from tensordict import TensorDict

from aioway.tables import Table, TensorDictListTable, TensorDictTable
from tests import fake


def block_table(device, batch_size, data_size) -> TensorDictTable:
    block = fake.tensordict_ok(size=data_size, device=device)
    return TensorDictTable(block, batch=batch_size)


def list_table(device, batch_size, data_size):
    return TensorDictListTable(
        [
            fake.tensordict_ok(size=batch_size, device=device)
            for _ in range(0, data_size, batch_size)
        ]
    )


@pytest.fixture(params=[block_table, list_table])
def table(
    request: pytest.FixtureRequest,
    device: str,
    batch_size: int,
    data_size: int,
) -> Table:
    return request.param(device=device, batch_size=batch_size, data_size=data_size)


def test_table_not_empty(table):
    assert table
    assert len(table)
    assert len(table) > 0


def test_table_idx_access(table):
    for idx in range(-len(table), len(table)):
        out = table[idx]
        assert isinstance(out, TensorDict)


def test_table_out_of_bounds(table):
    with pytest.raises(IndexError):
        _ = table[2 * len(table)]

    with pytest.raises(IndexError):
        _ = table[-2 * len(table)]


def test_table_convert_to_stream(table):
    list_table = list(table)
    list_stream = list(table.stream())

    assert len(list_table) == len(list_stream)

    for left, right in zip(list_table, list_stream):
        assert (left == right).all()
