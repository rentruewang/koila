# Copyright (c) RenChu Wang - All Rights Reserved

from collections import Counter

import pytest
import tensordict
from tensordict import TensorDict

from aioway.blocks import Block
from aioway.execs import FrameStreamExec, MatrixJoinExec, ZipExec
from aioway.frames import BlockFrame
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


def test_concat_exec_input_len(block_frame, concat_frame):
    assert len(block_frame) == len(concat_frame)


def test_concat_exec_next(block_frame, concat_frame, size):
    stream = ZipExec(
        FrameStreamExec.tabular(block_frame, {"batch_size": size}),
        FrameStreamExec.tabular(concat_frame, {"batch_size": size}),
    )

    for result, lhs, rhs in zip(
        stream,
        FrameStreamExec.tabular(block_frame, {"batch_size": size}),
        FrameStreamExec.tabular(concat_frame, {"batch_size": size}),
    ):
        concat = TensorDict({**lhs.data, **rhs.data}, device=result.data.device)
        assert (result.data == concat).all()


@pytest.fixture(scope="module")
def joinable_frame(device):
    block = fake.unionable_block_ok(size=max(fake.block_sizes()), device=device)
    return BlockFrame(block)


def test_join_exec_input_len(block_frame, joinable_frame):
    assert len(block_frame) * len(joinable_frame)


@pytest.fixture(scope="module")
def join_batch_size():
    return 64


def test_join_exec_next(block_frame, joinable_frame, join_batch_size):
    stream = MatrixJoinExec(
        FrameStreamExec.tabular(block_frame, {"batch_size": join_batch_size}),
        joinable_frame,
        on="i1d",
        rhs_batch=join_batch_size,
    )

    results: list[TensorDict] = []
    for rows in stream:
        assert isinstance(rows, Block), type(rows)
        results.append(rows.data)
    answer_items = tensordict.cat(results)["i1d"]

    answer_count = Counter(answer_items.tolist())
    left_count = Counter(block_frame.block.data["i1d"].tolist())
    right_count = Counter(joinable_frame.block.data["i1d"].tolist())

    assert answer_count.keys() == left_count.keys() & right_count.keys()

    for key in answer_count.keys():
        assert answer_count[key] == left_count[key] * right_count[key]


set.difference
