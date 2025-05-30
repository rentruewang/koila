# Copyright (c) AIoWay Authors - All Rights Reserved

import numpy as np
import pytest
import torch
from numpy import random as np_rand
from tensordict import TensorDict
from torch import Tensor

from aioway.blocks import Block

from . import fake


@pytest.fixture(params=fake.cpu_and_maybe_cuda(), scope="session")
def device(request) -> str:
    return request.param


@pytest.fixture(params=fake.block_sizes(), scope="module")
def batch(request) -> int:
    return request.param


def test_block_init_success(device, batch):
    _ = fake.block_ok(device=device, size=batch)


def test_block_init_fail_no_batch(device, batch):
    tensordict = fake.tensordict_no_batch(device=device, size=batch)

    with pytest.raises(IndexError):
        _ = Block(tensordict)


@pytest.mark.parametrize("data", fake.random_things())
def test_block_init_fail_not_tensordict(data):
    with pytest.raises(TypeError):
        _ = Block(data)


def test_block_len(device, batch):
    block = fake.block_ok(device=device, size=batch)
    assert len(block) == batch
    assert block.batch_size == (batch,)


def test_block_getitem_type(device, batch):
    block = fake.block_ok(device=device, size=batch)

    assert isinstance(block["f1d"], Tensor)
    assert isinstance(block["f2d"], Tensor)

    assert isinstance(block[0], Block)
    assert isinstance(block[1:2], Block)
    assert isinstance(block[[0]], Block)


def test_block_getitem_size(device, batch):
    block = fake.block_ok(device=device, size=batch)

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
    assert block[np_idx].batch_size == np_idx.sum()

    # Int index in numpy.
    assert block[np.arange(len(np_idx))[np_idx]].batch_size == np_idx.sum()


def test_block_device(device, batch):
    block = fake.block_ok(device=device, size=batch)
    assert block.device == device


def test_block_keys(device, batch):
    block = fake.block_ok(device=device, size=batch)
    assert block.keys() == {"f1d", "f2d", "i1d", "i2d"}


def rametest_block_rename(device, batch):
    block = fake.block_ok(device=device, size=batch)
    assert block.rename(f1d="f1", f2d="f2").keys() == {"f1", "f2", "i1d", "i2d"}


def test_block_chain(device, batch):
    block = fake.block_ok(device=device, size=batch)
    another = fake.unionable_block_ok(device=device, size=batch)

    chained = block.chain(another)
    assert len(chained) == len(block) + len(another)

    assert chained.keys() == block.keys() == another.keys()

    for key in chained.keys():
        ck = chained[key]
        bk = block[key]
        ak = another[key]
        assert (ck == torch.cat([bk, ak])).all()


def test_block_zip(device, batch):
    block = fake.block_ok(device=device, size=batch)
    another = fake.concat_block_ok(device=device, size=batch)

    concated = block.zip(another)
    assert len(concated) == len(block) == len(another)
    assert concated.keys() == {*block.keys(), *another.keys()}

    for idx in range(len(concated)):
        cd = concated[idx]
        bd = block[idx]
        ad = another[idx]

        assert (cd.data == TensorDict({**ad.data, **bd.data})).all()


def test_block_filter(device, batch):
    block = fake.block_ok(device=device, size=batch)

    f1d = block["f1d"]

    positive = (f1d > 0).cpu().numpy()

    golden_index = np.arange(len(block))[positive]
    golden = block[golden_index]

    filtered = block.filter("f1d > 0")

    assert (golden.data == filtered.data).all()
