# Copyright (c) AIoWay Authors - All Rights Reserved


import pytest

from aioway import _registries
from aioway.execs import Exec
from aioway.io import TorchFrame
from aioway.ops import Thunk
from tests import fake


@pytest.fixture(params=fake.cpu_and_maybe_cuda(), scope="module")
def device(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def data_size():
    return max(fake.batch_sizes())


@pytest.fixture(scope="module", params=fake.batch_sizes())
def batch_size(request) -> int:
    return request.param


@pytest.fixture(scope="module")
def block_frame(device, batch_size, data_size) -> TorchFrame:
    block = fake.tensordict_ok(size=data_size, device=device)
    return TorchFrame(block, batch=batch_size)


@pytest.fixture(scope="module")
def concat_frame(device, batch_size, data_size) -> TorchFrame:
    block = fake.concat_ok(size=data_size, device=device)
    return TorchFrame(block, batch_size)


@pytest.fixture(scope="module")
def joinable_frame(device, batch_size, data_size):
    block = fake.unionable_ok(size=data_size, device=device)
    return TorchFrame(block, batch_size)


def _exec_strat():
    yield "TREE"
    yield "DAG"


@pytest.fixture(params=_exec_strat())
def exec_strat(request) -> str:
    return request.param


@pytest.fixture
def make_executor(exec_strat):
    def executor(thunk: Thunk) -> Exec:
        reg = _registries.of(Exec)
        return reg[exec_strat](thunk)

    return executor
