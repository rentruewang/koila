# Copyright (c) AIoWay Authors - All Rights Reserved

"The shared utilities for `Stream` testing."

import pytest

from aioway import datasets
from tests import fake


@pytest.fixture
def block_table(device: str, data_size: int) -> datasets.ChunkFrame:
    block = fake.chunk_ok(size=data_size, device=device)
    return datasets.ChunkFrame(data=block)


@pytest.fixture
def table_stream(block_table: datasets.Frame, batch_size: int) -> datasets.FrameStream:
    return datasets.FrameStream(
        frame=block_table,
        options=datasets.FrameStreamLoader(batch_size=batch_size),
    )


@pytest.fixture
def concat_frame(device: str, data_size: int) -> datasets.ChunkFrame:
    block = fake.concat_ok(size=data_size, device=device)
    return datasets.ChunkFrame(data=block)


@pytest.fixture
def concat_stream(
    concat_frame: datasets.Frame, batch_size: int
) -> datasets.FrameStream:
    return datasets.FrameStream(
        frame=concat_frame,
        options=datasets.FrameStreamLoader(batch_size=batch_size),
    )


@pytest.fixture
def joinable_frame(device: str, data_size: int) -> datasets.ChunkFrame:
    "`datasets.Frame` for joining on the RHS."

    block = fake.unionable_ok(size=data_size, device=device)
    return datasets.ChunkFrame(data=block)


@pytest.fixture
def joinable_stream(
    joinable_frame: datasets.Frame, batch_size: int
) -> datasets.FrameStream:
    "`Stream` for joining on the RHS."
    return datasets.FrameStream(
        frame=joinable_frame,
        options=datasets.FrameStreamLoader(batch_size=batch_size),
    )
