# Copyright (c) AIoWay Authors - All Rights Reserved

import numpy as np
import pytest
from numpy import random

from aioway.batches import Chunk
from aioway.dsets import (
    ChunkFrame,
    ChunkListFrame,
    Frame,
    FrameStream,
    FrameStreamLoader,
)
from tests import fake


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
def frame(request, device, batch_size, data_size) -> Frame:
    return request.param(device=device, batch_size=batch_size, data_size=data_size)


@pytest.fixture
def table_stream(frame, batch_size):
    return FrameStream(
        frame,
        FrameStreamLoader(batch_size=batch_size),
    )


def test_table_not_empty(frame):
    assert frame
    assert len(frame)


def test_table_idx_arr(frame):
    idx = random.randint(low=-len(frame), high=len(frame), size=[len(frame)])

    assert np.all(-len(frame) <= idx)
    assert np.all(idx < len(frame))
    assert idx.shape == (len(frame),)

    out = frame[idx]
    assert isinstance(out, Chunk)
    assert len(out) == len(idx)


def test_table_idx_slice(frame):
    out = frame[-len(frame) : len(frame)]
    assert isinstance(out, Chunk)
    assert len(out) == len(frame)


def test_table_out_of_bounds(frame):
    with pytest.raises(IndexError):
        _ = frame[[-2 * len(frame)]]
