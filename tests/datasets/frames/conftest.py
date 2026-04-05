# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway.datasets import (
    ChunkFrame,
    ChunkListFrame,
    Frame,
    FrameStream,
    FrameStreamLoader,
)


def block_table(device: str, batch_size: int, data_size: int):
    block = fake.chunk_ok(size=data_size, device=device)
    return ChunkFrame(block)


def list_table(device: str, batch_size: int, data_size: int):
    return ChunkListFrame(
        [
            fake.chunk_ok(size=batch_size, device=device)
            for _ in range(0, data_size, batch_size)
        ]
    )


@pytest.fixture(params=[block_table, list_table])
def frame(
    request: pytest.FixtureRequest, device: str, batch_size: int, data_size: int
) -> Frame:
    return request.param(device=device, batch_size=batch_size, data_size=data_size)


@pytest.fixture
def table_stream(frame: Frame, batch_size: int):
    return FrameStream(
        frame,
        FrameStreamLoader(batch_size=batch_size),
    )
