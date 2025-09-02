# Copyright (c) AIoWay Authors - All Rights Reserved

import random
from collections.abc import Callable

import pytest

from aioway.execs import (
    Exec,
    ExprFilterExec,
    FrameExec,
    FuncFilterExec,
    MapExec,
    ProjectExec,
    RenameExec,
)
from aioway.frames import BlockFrame
from tests import fake


@pytest.fixture(params=fake.cpu_and_maybe_cuda(), scope="module")
def device(request) -> str:
    return request.param


@pytest.fixture(params=fake.block_sizes(), scope="module")
def size(request) -> int:
    return request.param


@pytest.fixture(scope="module")
def block_frame(device: str):
    block = fake.block_ok(size=max(fake.block_sizes()), device=device)
    return BlockFrame(block)


def test_block_frame_len(block_frame):
    assert len(block_frame) == max(fake.block_sizes())


def test_block_frame_getitem(block_frame):
    idx = random.randrange(len(block_frame))
    assert isinstance(idx, int)

    assert (block_frame[idx].data == block_frame.block.data[idx]).all()
    assert not block_frame[idx].batch_dims


@pytest.fixture
def block_frame_iter(block_frame, size):
    # Note:
    #   Do not use this iterator in tests for comparison,
    #   because the iterator is instantiated once for each tests.
    #   Let's look at an example:
    #   >>> to_test = PassThroughStream(iter)
    #   >>> golden = iter
    #   >>> for a, b in zip(to_test, golden):
    #   >>>    # This would fail!
    #   >>>    assert a is not b
    #   Both sides of the equation uses the same iterator underneath,
    #   and it would create subtle bugs.
    return FrameExec(dataset=block_frame, opt={"batch_size": size})


def test_iterator_exec(block_frame_iter):
    assert isinstance(block_frame_iter, Exec)


def test_iterator_eq(block_frame, size):
    for fresh, iterator in zip(
        FrameExec(dataset=block_frame, opt={"batch_size": size}),
        FrameExec(dataset=block_frame, opt={"batch_size": size}),
    ):
        assert (fresh.data == iterator.data).all()


def filter_expr_exec(stream: Exec):
    return ExprFilterExec(stream, expr="f1d > 0")


def filter_pred_frame(stream: Exec):
    return FuncFilterExec(stream, predicate=lambda t: (t["f1d"] > 0).cpu().numpy())


@pytest.fixture(params=[filter_expr_exec, filter_pred_frame])
def filter_exec(request) -> Callable[[Exec], Exec]:
    return request.param


def test_filter_exec_next(filter_exec, block_frame, size):
    frame_exec_left = FrameExec(dataset=block_frame, opt={"batch_size": size})
    frame_exec_right = FrameExec(dataset=block_frame, opt={"batch_size": size})
    stream = filter_exec(frame_exec_left)
    for filtered, original in zip(stream, frame_exec_right):
        assert (filtered.data == original.filter("f1d > 0").data).all()


@pytest.fixture(scope="module")
def rename_op():
    return {"f1d": "f1", "f2d": "f2", "i1d": "i1", "i2d": "i2"}


def test_rename_exec_next(block_frame, size, rename_op):
    exec_left = FrameExec(dataset=block_frame, opt={"batch_size": size})
    exec_right = FrameExec(dataset=block_frame, opt={"batch_size": size})

    for renamed, original in zip(RenameExec(exec_left, renames=rename_op), exec_right):
        assert (
            renamed.data == original.rename(f1d="f1", f2d="f2", i1d="i1", i2d="i2").data
        ).all()


@pytest.fixture
def map_rename_op():
    return {"f1d": "f", "i1d": "i"}


@pytest.fixture
def map_exec(block_frame, map_rename_op, size):
    frame_exec = FrameExec(dataset=block_frame, opt={"batch_size": size})
    return MapExec(
        frame_exec,
        compute=lambda b: b.rename(**map_rename_op),
    )


def test_map_exec_next(map_exec, block_frame, map_rename_op, size):
    for mapped, original in zip(
        map_exec, FrameExec(dataset=block_frame, opt={"batch_size": size})
    ):
        assert (mapped.data == original.rename(**map_rename_op).data).all()


def test_project_exec_next(block_frame, size):
    exec_left = FrameExec(dataset=block_frame, opt={"batch_size": size})
    exec_right = FrameExec(dataset=block_frame, opt={"batch_size": size})
    project_exec = ProjectExec(exec_left, subset=["f1d", "i2d"])
    for curr, other in zip(project_exec, exec_right):
        assert (curr.data == other[["f1d", "i2d"]].data).all()
