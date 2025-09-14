# Copyright (c) AIoWay Authors - All Rights Reserved


import pytest

from aioway import execs
from aioway.execs import Exec
from aioway.frames import BlockFrame
from aioway.ops import FrameOp, Thunk
from tests import fake


@pytest.fixture(params=fake.cpu_and_maybe_cuda(), scope="module")
def device(request) -> str:
    return request.param


@pytest.fixture(params=fake.block_sizes(), scope="module")
def size(request) -> int:
    return request.param


@pytest.fixture(scope="module")
def block_frame(device) -> BlockFrame:
    block = fake.block_ok(size=max(fake.block_sizes()), device=device)
    return BlockFrame(block)


@pytest.fixture(scope="module")
def concat_frame(device) -> BlockFrame:
    block = fake.concat_block_ok(size=max(fake.block_sizes()), device=device)
    return BlockFrame(block)


@pytest.fixture(scope="module")
def joinable_frame(device):
    block = fake.unionable_block_ok(size=max(fake.block_sizes()), device=device)
    return BlockFrame(block)


@pytest.fixture
def frame_op_loader_cfg(size):
    from aioway.ops.nullary import FrameDataLoaderCfg

    return FrameDataLoaderCfg(batch_size=size, pin_memory=True)


@pytest.fixture
def concat_frame_op(concat_frame, frame_op_loader_cfg):

    return FrameOp(concat_frame, frame_op_loader_cfg)


@pytest.fixture
def block_frame_op(block_frame, frame_op_loader_cfg):
    return FrameOp(block_frame, frame_op_loader_cfg)


@pytest.fixture
def joinable_frame_op(joinable_frame, frame_op_loader_cfg):
    return FrameOp(joinable_frame, frame_op_loader_cfg)


@pytest.fixture
def another_block_frame_op(block_frame, frame_op_loader_cfg):
    return FrameOp(block_frame, frame_op_loader_cfg)


def _exec_strat():
    yield "LAZY"
    yield "DAG"


@pytest.fixture(params=_exec_strat())
def exec_strat(request):
    return request.param


@pytest.fixture
def make_executor(exec_strat):
    def executor(thunk: Thunk) -> Exec:
        return execs.execute(thunk=thunk, strategy=exec_strat)

    return executor
