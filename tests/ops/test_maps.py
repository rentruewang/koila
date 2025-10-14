# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway.plans import FuncPlan, Plan, ProjectPlan, RenamePlan, _funcs
from aioway.schedulers import Scheduler
from aioway.tables import Table


def test_filter(filterer: Plan, block_frame: Table, scheduler: Scheduler):
    for filtered, original in zip(
        scheduler(filterer(block_frame.plan())),
        scheduler(block_frame.plan()),
    ):
        original_filtered = _funcs.filter(original, "f1d > 0")
        assert (filtered == original_filtered).all()


@pytest.fixture(scope="module")
def renames():
    return {"f1d": "f1", "f2d": "f2", "i1d": "i1", "i2d": "i2"}


def test_rename(block_frame: Table, renames: dict[str, str], scheduler: Scheduler):
    rename_op = RenamePlan(renames)
    for renamed, original in zip(
        scheduler(rename_op(block_frame.plan())),
        scheduler(block_frame.plan()),
    ):
        original_renamed = _funcs.rename(
            original, f1d="f1", f2d="f2", i1d="i1", i2d="i2"
        )
        assert (renamed == original_renamed).all()


@pytest.fixture
def map_rename():
    return {"f1d": "f", "i1d": "i"}


def test_func_op(block_frame: Table, map_rename: dict[str, str], scheduler: Scheduler):
    f = lambda b: _funcs.rename(b, **map_rename)
    func_plan = FuncPlan(func=f)
    for mapped, original in zip(
        scheduler(func_plan(block_frame.plan())),
        scheduler(block_frame.plan()),
    ):
        assert (mapped == f(original)).all()


def test_project(block_frame: Table, scheduler: Scheduler):
    project_plan = ProjectPlan(subset=["f1d", "i2d"])
    for curr, other in zip(
        scheduler(project_plan(block_frame.plan())),
        scheduler(block_frame.plan()),
    ):
        assert (curr == other.select("f1d", "i2d")).all()
