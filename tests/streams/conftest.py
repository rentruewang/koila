# Copyright (c) AIoWay Authors - All Rights Reserved

"The shared utilities for ``Stream`` testing."


import pytest

from aioway.tables import TableStream, TensorDictTable
from tests import fake


@pytest.fixture
def block_table(device, batch_size, data_size) -> TensorDictTable:
    block = fake.tensordict_ok(size=data_size, device=device)
    return TensorDictTable(block, batch=batch_size)


@pytest.fixture
def table_stream(block_table: TensorDictTable) -> TableStream:
    return TableStream(table=block_table)


@pytest.fixture
def concat_table(device: str, batch_size: int, data_size: int) -> TensorDictTable:
    block = fake.concat_ok(size=data_size, device=device)
    return TensorDictTable(source=block, batch=batch_size)


@pytest.fixture
def concat_stream(concat_table: TensorDictTable):
    return TableStream(table=concat_table)


@pytest.fixture
def joinable_table(device: str, batch_size: int, data_size: int) -> TensorDictTable:
    "Table for joining on the RHS."
    block = fake.unionable_ok(size=data_size, device=device)
    return TensorDictTable(source=block, batch=batch_size)


@pytest.fixture
def joinable_stream(joinable_table: TensorDictTable):
    "``Stream`` for joining on the RHS."
    return TableStream(table=joinable_table)
