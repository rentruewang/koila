# Copyright (c) AIoWay Authors - All Rights Reserved

import numpy as np
import pytest
from numpy import random

from aioway.batches import Chunk
from aioway.dsets import (
    ChunkListTable,
    ChunkTable,
    Table,
    TableStream,
    TableStreamLoader,
)
from tests import fake


def block_table(device: str, batch_size: int, data_size: int) -> ChunkTable:
    block = fake.chunk_ok(size=data_size, device=device)
    return ChunkTable(block)


def list_table(device: str, batch_size: int, data_size: int):
    return ChunkListTable(
        [
            fake.chunk_ok(size=batch_size, device=device)
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


@pytest.fixture
def table_stream(table: Table, batch_size: int):
    return TableStream(
        table,
        TableStreamLoader(batch_size=batch_size),
    )


def test_table_not_empty(table: Table):
    assert table
    assert len(table)


def test_table_idx_arr(table: Table):
    idx = random.randint(low=-len(table), high=len(table), size=[len(table)])

    assert np.all(-len(table) <= idx)
    assert np.all(idx < len(table))
    assert idx.shape == (len(table),)

    out = table[idx]
    assert isinstance(out, Chunk)
    assert len(out) == len(idx)


def test_table_idx_slice(table: Table):
    out = table[-len(table) : len(table)]
    assert isinstance(out, Chunk)
    assert len(out) == len(table)


def test_table_out_of_bounds(table: Table):
    with pytest.raises(IndexError):
        _ = table[[2 * len(table)]]

    with pytest.raises(IndexError):
        _ = table[[-2 * len(table)]]
