# Copyright (c) AIoWay Authors - All Rights Reserved


import pytest

from aioway import schedulers
from aioway.plans import (
    ExprFilterPlan,
    FuncFilterPlan,
    MatchPlan,
    Plan1,
    RepeatPlan,
    ZipPlan,
)
from aioway.schedulers import Scheduler
from aioway.tables import TorchTable
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
def block_frame(device, batch_size, data_size) -> TorchTable:
    block = fake.tensordict_ok(size=data_size, device=device)
    return TorchTable(block, batch=batch_size)


@pytest.fixture(scope="module")
def concat_frame(device, batch_size, data_size) -> TorchTable:
    block = fake.concat_ok(size=data_size, device=device)
    return TorchTable(block, batch_size)


@pytest.fixture(scope="module")
def joinable_frame(device, batch_size, data_size):
    block = fake.unionable_ok(size=data_size, device=device)
    return TorchTable(block, batch_size)


def _exec_strat():
    yield schedulers.tree
    yield schedulers.dag


@pytest.fixture(params=_exec_strat())
def exec_strat(request) -> Scheduler:
    return request.param


@pytest.fixture
def scheduler(exec_strat: Scheduler) -> Scheduler:
    return exec_strat


@pytest.fixture
def matcher():
    return MatchPlan(key="i1d")


@pytest.fixture
def zipper():
    return ZipPlan()


@pytest.fixture
def repeater(times):
    return RepeatPlan(times=times)


def _filter_expr_exec():
    return ExprFilterPlan("f1d > 0")


def _filter_pred_frame():
    return FuncFilterPlan(predicate=lambda t: (t["f1d"] > 0).cpu())


@pytest.fixture(params=[_filter_expr_exec, _filter_pred_frame])
def filterer(request) -> Plan1:
    return request.param()
