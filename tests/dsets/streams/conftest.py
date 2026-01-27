# Copyright (c) AIoWay Authors - All Rights Reserved

"The shared utilities for ``Stream`` testing."


import pytest

from aioway.dsets import TableStream, TableStreamLoader, TensorDictTable
from tests import fake


@pytest.fixture
def block_table(device: str, data_size: int) -> TensorDictTable:
    block = fake.tensordict_ok(size=data_size, device=device)
    return TensorDictTable(data=block)


@pytest.fixture
def table_stream(block_table: TensorDictTable, batch_size: int) -> TableStream:
    return TableStream(
        table=block_table,
        options=TableStreamLoader(batch_size=batch_size),
    )


@pytest.fixture
def concat_table(device: str, data_size: int) -> TensorDictTable:
    block = fake.concat_ok(size=data_size, device=device)
    return TensorDictTable(data=block)


@pytest.fixture
def concat_stream(concat_table: TensorDictTable, batch_size: int):
    return TableStream(
        table=concat_table,
        options=TableStreamLoader(batch_size=batch_size),
    )


@pytest.fixture
def joinable_table(device: str, data_size: int) -> TensorDictTable:
    "Table for joining on the RHS."
    block = fake.unionable_ok(size=data_size, device=device)
    return TensorDictTable(data=block)


@pytest.fixture
def joinable_stream(joinable_table: TensorDictTable, batch_size: int):
    "``Stream`` for joining on the RHS."
    return TableStream(
        table=joinable_table,
        options=TableStreamLoader(batch_size=batch_size),
    )
