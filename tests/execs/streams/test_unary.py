# Copyright (c) RenChu Wang - All Rights Reserved

import math
import random
from collections.abc import Callable

import pytest

from aioway.execs import (
    Exec,
    FilterExprExec,
    FilterPredExec,
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
def block_frame(size: int, device: str):
    block = fake.block_ok(size=max(fake.block_sizes()), device=device)
    return BlockFrame(block, max_batch=size)


def test_block_frame_len(block_frame):
    assert len(block_frame) == math.ceil(
        max(fake.block_sizes()) / block_frame.max_batch
    )


def test_block_frame_getitem(block_frame):
    idx = random.randrange(len(block_frame))

    start, stop = block_frame.max_batch * idx, block_frame.max_batch * (idx + 1)
    assert (block_frame[idx].data == block_frame.block.data[start:stop]).all()

    assert len(block_frame[idx]) == (
        block_frame.max_batch
        if stop < len(block_frame.block.data)
        else len(block_frame.block.data) - start
    )


@pytest.fixture
def block_frame_iter(block_frame) -> Exec:
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
    return iter(block_frame)


def test_iterator_stream(block_frame_iter):
    assert isinstance(block_frame_iter, Exec)


def test_iterator_eq(block_frame, block_frame_iter):
    for fresh, iterator in zip(block_frame, block_frame_iter):
        assert (fresh.data == iterator.data).all()


def filter_expr_stream(stream: Exec):
    return FilterExprExec(stream, "f1d > 0")


def filter_pred_frame(stream: Exec):
    return FilterPredExec(stream, predicate=lambda t: (t["f1d"] > 0).cpu().numpy())


@pytest.fixture(params=[filter_expr_stream, filter_pred_frame])
def filter_stream(request) -> Callable[[Exec], Exec]:
    return request.param


def test_filter_stream_attrs(filter_stream, block_frame):
    assert filter_stream(iter(block_frame)).attrs == block_frame.attrs


def test_filter_stream_next(filter_stream, block_frame):
    stream = filter_stream(iter(block_frame))
    for filtered, original in zip(stream, block_frame):
        assert (filtered.data == original.filter("f1d > 0").data).all()


@pytest.fixture(scope="module")
def rename_op():
    return {"f1d": "f1", "f2d": "f2", "i1d": "i1", "i2d": "i2"}


@pytest.fixture
def rename_stream(block_frame, rename_op):
    return RenameExec(iter(block_frame), **rename_op)


def test_rename_stream_attrs(rename_stream, block_frame, rename_op):
    assert rename_stream.attrs == block_frame.attrs.rename(**rename_op)


def test_rename_stream_next(rename_stream, block_frame):
    for renamed, original in zip(rename_stream, block_frame):
        assert (
            renamed.data == original.rename(f1d="f1", f2d="f2", i1d="i1", i2d="i2").data
        ).all()


@pytest.fixture
def map_rename_op():
    return {"f1d": "f", "i1d": "i"}


@pytest.fixture
def map_stream(block_frame, map_rename_op):
    return MapExec(
        iter(block_frame),
        lambda b: b.rename(**map_rename_op),
        output=block_frame.attrs.rename(**map_rename_op),
    )


def test_map_stream_next(map_stream, block_frame, map_rename_op):
    for mapped, original in zip(map_stream, block_frame):
        assert (mapped.data == original.rename(**map_rename_op).data).all()


@pytest.fixture
def project_stream(block_frame):
    return ProjectExec(iter(block_frame), subset=["f1d", "i2d"])


def test_project_stream_attrs(project_stream, block_frame):
    selected = {key: block_frame.attrs[key] for key in ["f1d", "i2d"]}
    assert project_stream.attrs == selected


def test_project_stream_next(project_stream, block_frame):
    for curr, other in zip(project_stream, block_frame):
        assert (curr.data == other[["f1d", "i2d"]].data).all()
