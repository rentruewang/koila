# Copyright (c) AIoWay Authors - All Rights Reserved

import random
from collections.abc import Callable

import pytest

from aioway.blocks import Block
from aioway.execs import (
    EchoExec,
    Execution,
    FilterExprExec,
    FilterPredExec,
    FrameExec,
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
def block_frame_iter(block_frame, size) -> Execution:
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
    return FrameExec(block_frame, {"batch_size": size})


def test_iterator_exec(block_frame_iter):
    assert isinstance(block_frame_iter, Execution)


def test_iterator_eq(block_frame, block_frame_iter, size):
    for fresh, iterator in zip(
        FrameExec(block_frame, {"batch_size": size}), block_frame_iter
    ):
        assert (fresh.data == iterator.data).all()


def filter_expr_exec(stream: Execution):
    return FilterExprExec(stream, "f1d > 0")


def filter_pred_frame(stream: Execution):
    return FilterPredExec(stream, predicate=lambda t: (t["f1d"] > 0).cpu().numpy())


@pytest.fixture(params=[filter_expr_exec, filter_pred_frame])
def filter_exec(request) -> Callable[[Execution], Execution]:
    return request.param


def test_filter_exec_attrs(filter_exec, block_frame, size):
    assert (
        filter_exec(FrameExec(block_frame, {"batch_size": size})).attrs
        == block_frame.attrs
    )


def test_filter_exec_next(filter_exec, block_frame, size):
    stream = filter_exec(FrameExec(block_frame, {"batch_size": size}))
    for filtered, original in zip(stream, FrameExec(block_frame, {"batch_size": size})):
        assert (filtered.data == original.filter("f1d > 0").data).all()


@pytest.fixture(scope="module")
def rename_op():
    return {"f1d": "f1", "f2d": "f2", "i1d": "i1", "i2d": "i2"}


@pytest.fixture
def rename_exec(block_frame, rename_op, size):
    return RenameExec(FrameExec(block_frame, {"batch_size": size}), **rename_op)


def test_rename_exec_attrs(rename_exec, block_frame, rename_op):
    assert rename_exec.attrs == block_frame.attrs.rename(**rename_op)


def test_rename_exec_next(rename_exec, block_frame, size):
    for renamed, original in zip(
        rename_exec, FrameExec(block_frame, {"batch_size": size})
    ):
        assert (
            renamed.data == original.rename(f1d="f1", f2d="f2", i1d="i1", i2d="i2").data
        ).all()


@pytest.fixture
def map_rename_op():
    return {"f1d": "f", "i1d": "i"}


@pytest.fixture
def map_exec(block_frame, map_rename_op, size):
    return MapExec(
        child=FrameExec(block_frame, {"batch_size": size}),
        compute=lambda b: b.rename(**map_rename_op),
        output=block_frame.attrs.rename(**map_rename_op),
    )


def test_map_exec_next(map_exec, block_frame, map_rename_op, size):
    for mapped, original in zip(map_exec, FrameExec(block_frame, {"batch_size": size})):
        assert (mapped.data == original.rename(**map_rename_op).data).all()


@pytest.fixture
def project_exec(block_frame, size):
    return ProjectExec(
        FrameExec(block_frame, {"batch_size": size}),
        subset=["f1d", "i2d"],
    )


def test_project_exec_attrs(project_exec, block_frame):
    selected = {key: block_frame.attrs[key] for key in ["f1d", "i2d"]}
    assert project_exec.attrs == selected


def test_project_exec_next(project_exec, block_frame, size):
    for curr, other in zip(project_exec, FrameExec(block_frame, {"batch_size": size})):
        assert (curr.data == other[["f1d", "i2d"]].data).all()


@pytest.fixture
def echo_times():
    return 3


@pytest.fixture
def echo_exec(block_frame, size, echo_times):
    return EchoExec(FrameExec(block_frame, {"batch_size": size}), times=echo_times)


def test_echo_exec_next(echo_exec, echo_times):
    while True:
        execs = []

        for i in range(echo_times):
            try:
                execs.append(next(echo_exec))
            except StopIteration:
                assert i == 0, (
                    f"`EchoExec` should yield exactly {echo_times} times of the same block, "
                    f"but exited after {i+1} iteration"
                )
                return

        assert len(execs) == echo_times
        assert all(
            isinstance(e, Block) for e in execs
        ), "`EchoExec` must yield `Block`s."
        assert (
            len({id(e) for e in execs}) == 1
        ), f"`EchoExec` should yield the same object {echo_times} times."
