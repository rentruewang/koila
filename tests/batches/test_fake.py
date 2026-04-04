# Copyright (c) AIoWay Authors - All Rights Reserved

import numpy as np
import pytest
import torch
from numpy import random as np_rand

from aioway import chunks
from tests import fake


@pytest.fixture(params=fake.cpu_and_maybe_cuda(), scope="session")
def device(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(params=fake.batch_sizes(), scope="module")
def batch(request: pytest.FixtureRequest) -> int:
    return request.param


def test_chunk_init_success(device: str, batch: int) -> None:
    _ = fake.chunk_ok(device=device, size=batch)


def test_chunk_len(device: str, batch: int) -> None:
    block = fake.chunk_ok(device=device, size=batch)
    assert len(block) == batch


def test_chunk_getitem_size(device: str, batch: int) -> None:
    block = fake.chunk_ok(device=device, size=batch)

    assert len(block[batch - 1 : batch]) == 1
    assert len(block[[0]]) == 1
    assert len(block[[-1]]) == 1

    # Bool index in torch.
    torch_idx = torch.randn(batch) > 0
    assert len(block[torch_idx]) == (torch_idx > 0).sum()

    # Int index in torch.
    indexed: chunks.Chunk = block[torch.arange(len(torch_idx))[torch_idx]]
    assert len(indexed) == (torch_idx > 0).sum().item()

    # Bool index in numpy.
    np_idx = np_rand.randn(batch) < 0
    assert len(block[torch.tensor(np_idx)]) == np_idx.sum()

    # Int index in numpy.
    assert len(block[np.arange(len(np_idx))[np_idx]]) == np_idx.sum()


def test_chunk_keys(device: str, batch: int) -> None:
    block = fake.chunk_ok(device=device, size=batch)
    assert set(block.keys()) == {"f1d", "f2d", "i1d", "i2d"}
