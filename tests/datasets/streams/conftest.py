# Copyright (c) AIoWay Authors - All Rights Reserved

"The shared utilities for `Stream` testing."

import pytest

from aioway.datasets import ChunkFrame, FrameStream, FrameStreamLoader
from tests import fake


@pytest.fixture
def block_table(device, data_size) -> ChunkFrame:
    block = fake.chunk_ok(size=data_size, device=device)
    return ChunkFrame(data=block)


@pytest.fixture
def table_stream(block_table, batch_size) -> FrameStream:
    return FrameStream(
        frame=block_table,
        options=FrameStreamLoader(batch_size=batch_size),
    )


@pytest.fixture
def concat_frame(device, data_size) -> ChunkFrame:
    block = fake.concat_ok(size=data_size, device=device)
    return ChunkFrame(data=block)


@pytest.fixture
def concat_stream(concat_frame, batch_size) -> FrameStream:
    return FrameStream(
        frame=concat_frame,
        options=FrameStreamLoader(batch_size=batch_size),
    )


@pytest.fixture
def joinable_frame(device, data_size) -> ChunkFrame:
    "`Frame` for joining on the RHS."

    block = fake.unionable_ok(size=data_size, device=device)
    return ChunkFrame(data=block)


@pytest.fixture
def joinable_stream(joinable_frame, batch_size) -> FrameStream:
    "`Stream` for joining on the RHS."
    return FrameStream(
        frame=joinable_frame,
        options=FrameStreamLoader(batch_size=batch_size),
    )
