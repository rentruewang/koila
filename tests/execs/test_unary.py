# Copyright (c) RenChu Wang - All Rights Reserved

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
from aioway.tabular import BlockFrame
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
def block_frame_iter(block_frame, size) -> Exec:
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
    return block_frame.iterate(batch_size=size)


def test_iterator_stream(block_frame_iter):
    assert isinstance(block_frame_iter, Exec)


def test_iterator_eq(block_frame, block_frame_iter, size):
    for fresh, iterator in zip(block_frame.iterate(batch_size=size), block_frame_iter):
        assert (fresh.data == iterator.data).all()


def filter_expr_stream(stream: Exec):
    return FilterExprExec(stream, "f1d > 0")


def filter_pred_frame(stream: Exec):
    return FilterPredExec(stream, predicate=lambda t: (t["f1d"] > 0).cpu().numpy())


@pytest.fixture(params=[filter_expr_stream, filter_pred_frame])
def filter_stream(request) -> Callable[[Exec], Exec]:
    return request.param


def test_filter_stream_attrs(filter_stream, block_frame, size):
    assert (
        filter_stream(block_frame.iterate(batch_size=size)).attrs == block_frame.attrs
    )


def test_filter_stream_next(filter_stream, block_frame, size):
    stream = filter_stream(block_frame.iterate(batch_size=size))
    for filtered, original in zip(stream, block_frame.iterate(batch_size=size)):
        assert (filtered.data == original.filter("f1d > 0").data).all()


@pytest.fixture(scope="module")
def rename_op():
    return {"f1d": "f1", "f2d": "f2", "i1d": "i1", "i2d": "i2"}


@pytest.fixture
def rename_stream(block_frame, rename_op, size):
    return RenameExec(block_frame.iterate(batch_size=size), **rename_op)


def test_rename_stream_attrs(rename_stream, block_frame, rename_op):
    assert rename_stream.attrs == block_frame.attrs.rename(**rename_op)


def test_rename_stream_next(rename_stream, block_frame, size):
    for renamed, original in zip(rename_stream, block_frame.iterate(batch_size=size)):
        assert (
            renamed.data == original.rename(f1d="f1", f2d="f2", i1d="i1", i2d="i2").data
        ).all()


@pytest.fixture
def map_rename_op():
    return {"f1d": "f", "i1d": "i"}


@pytest.fixture
def map_stream(block_frame, map_rename_op, size):
    return MapExec(
        block_frame.iterate(batch_size=size),
        lambda b: b.rename(**map_rename_op),
        output=block_frame.attrs.rename(**map_rename_op),
    )


def test_map_stream_next(map_stream, block_frame, map_rename_op, size):
    for mapped, original in zip(map_stream, block_frame.iterate(batch_size=size)):
        assert (mapped.data == original.rename(**map_rename_op).data).all()


@pytest.fixture
def project_stream(block_frame, size):
    return ProjectExec(block_frame.iterate(batch_size=size), subset=["f1d", "i2d"])


def test_project_stream_attrs(project_stream, block_frame):
    selected = {key: block_frame.attrs[key] for key in ["f1d", "i2d"]}
    assert project_stream.attrs == selected


def test_project_stream_next(project_stream, block_frame, size):
    for curr, other in zip(project_stream, block_frame.iterate(batch_size=size)):
        assert (curr.data == other[["f1d", "i2d"]].data).all()
