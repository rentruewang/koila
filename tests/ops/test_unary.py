# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway.ops import (
    ExprFilterOp,
    FuncFilterOp,
    MapOp,
    Op,
    ProjectOp,
    RenameOp,
)


def filter_expr_exec():
    return ExprFilterOp("f1d > 0")


def filter_pred_frame():
    return FuncFilterOp(predicate=lambda t: (t["f1d"] > 0).cpu().numpy())


@pytest.fixture(params=[filter_expr_exec, filter_pred_frame])
def filter_op(request) -> Op:
    return request.param()


def test_filter(filter_op, block_frame_op):
    for filtered, original in zip(
        filter_op.thunk(block_frame_op.thunk()), block_frame_op.thunk()
    ):
        assert (filtered.data == original.filter("f1d > 0").data).all()


@pytest.fixture(scope="module")
def renames():
    return {"f1d": "f1", "f2d": "f2", "i1d": "i1", "i2d": "i2"}


def test_rename_exec_next(block_frame_op, renames):
    rename_op = RenameOp(renames)
    for renamed, original in zip(
        rename_op.thunk(block_frame_op.thunk()), block_frame_op.thunk()
    ):
        assert (
            renamed.data == original.rename(f1d="f1", f2d="f2", i1d="i1", i2d="i2").data
        ).all()


@pytest.fixture
def map_rename():
    return {"f1d": "f", "i1d": "i"}


def test_map_op(block_frame_op, map_rename):
    map_op = MapOp(compute=lambda b: b.rename(**map_rename))
    for mapped, original in zip(
        map_op.thunk(block_frame_op.thunk()), block_frame_op.thunk()
    ):
        assert (mapped.data == original.rename(**map_rename).data).all()


def test_project_exec_next(block_frame_op):
    project_op = ProjectOp(subset=["f1d", "i2d"])
    for curr, other in zip(
        project_op.thunk(block_frame_op.thunk()), block_frame_op.thunk()
    ):
        assert (curr.data == other[["f1d", "i2d"]].data).all()
