# Copyright (c) AIoWay Authors - All Rights Reserved

import numpy as np
import pytest
import torch
from numpy import random as np_rand

from aioway.ops import _funcs

from . import fake


@pytest.fixture(params=fake.cpu_and_maybe_cuda(), scope="session")
def device(request) -> str:
    return request.param


@pytest.fixture(params=fake.batch_sizes(), scope="module")
def batch(request) -> int:
    return request.param


def test_tensordict_init_success(device, batch):
    _ = fake.tensordict_ok(device=device, size=batch)


def test_tensordict_len(device, batch):
    block = fake.tensordict_ok(device=device, size=batch)
    assert len(block) == batch
    assert block.batch_size == (batch,)


def test_tensordict_getitem_size(device, batch):
    block = fake.tensordict_ok(device=device, size=batch)

    assert block[batch - 1 : batch].batch_size == (1,)
    assert block[[0]].batch_size == (1,)
    assert block[[-1]].batch_size == (1,)

    # Bool index in torch.
    torch_idx = torch.randn(batch) > 0
    assert block[torch_idx].batch_size == ((torch_idx > 0).sum().item(),)

    # Int index in torch.
    assert block[torch.arange(len(torch_idx))[torch_idx]].batch_size == (
        (torch_idx > 0).sum().item(),
    )

    # Bool index in numpy.
    np_idx = np_rand.randn(batch) < 0
    assert block[torch.tensor(np_idx)].batch_size == np_idx.sum()

    # Int index in numpy.
    assert block[np.arange(len(np_idx))[np_idx]].batch_size == np_idx.sum()


def test_tensordict_keys(device, batch):
    block = fake.tensordict_ok(device=device, size=batch)
    assert block.keys() == {"f1d", "f2d", "i1d", "i2d"}


def test_tensordict_filter(device, batch):
    block = fake.tensordict_ok(device=device, size=batch)

    f1d = block["f1d"]

    positive = (f1d > 0).cpu().numpy()

    golden_index = np.arange(len(block))[positive]
    golden = block[golden_index]

    filtered = _funcs.filter(block, "f1d > 0")

    assert (golden.data == filtered.data).all()
