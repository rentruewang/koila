# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway.rels import FuncPlan, ProjectPlan, RenamePlan
from aioway.rels.plans import _funcs


def test_filter(filter_op, block_frame, make_executor):
    for filtered, original in zip(
        make_executor(filter_op.thunk(block_frame.op.thunk())),
        make_executor(block_frame.op.thunk()),
    ):
        original_filtered = _funcs.filter(original, "f1d > 0")
        assert (filtered == original_filtered).all()


@pytest.fixture(scope="module")
def renames():
    return {"f1d": "f1", "f2d": "f2", "i1d": "i1", "i2d": "i2"}


def test_rename(block_frame, renames, make_executor):
    rename_op = RenamePlan(renames)
    for renamed, original in zip(
        make_executor(rename_op.thunk(block_frame.op.thunk())),
        make_executor(block_frame.op.thunk()),
    ):
        original_renamed = _funcs.rename(
            original, f1d="f1", f2d="f2", i1d="i1", i2d="i2"
        )
        assert (renamed == original_renamed).all()


@pytest.fixture
def map_rename():
    return {"f1d": "f", "i1d": "i"}


def test_func_op(block_frame, map_rename, make_executor):
    f = lambda b: _funcs.rename(b, **map_rename)
    func_op = FuncPlan(func=f)
    for mapped, original in zip(
        make_executor(func_op.thunk(block_frame.op.thunk())),
        make_executor(block_frame.op.thunk()),
    ):
        assert (mapped == f(original)).all()


def test_project(block_frame, make_executor):
    project_op = ProjectPlan(subset=["f1d", "i2d"])
    for curr, other in zip(
        make_executor(project_op.thunk(block_frame.op.thunk())),
        make_executor(block_frame.op.thunk()),
    ):
        assert (curr == other.select("f1d", "i2d")).all()
