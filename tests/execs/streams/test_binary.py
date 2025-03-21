# Copyright (c) RenChu Wang - All Rights Reserved

from collections import Counter

import pytest
from tensordict import TensorDict

from aioway.execs import MatrixJoinExec, ZipExec
from aioway.frames import BlockFrame
from tests import fake


@pytest.fixture(params=fake.cpu_and_maybe_cuda(), scope="module")
def device(request) -> str:
    return request.param


@pytest.fixture(params=fake.block_sizes(), scope="module")
def size(request) -> int:
    return request.param


@pytest.fixture(scope="module")
def block_frame(size, device):
    block = fake.block_ok(size=max(fake.block_sizes()), device=device)
    return BlockFrame(block, max_batch=size)


@pytest.fixture(scope="module")
def concat_frame(size, device):
    block = fake.concat_block_ok(size=max(fake.block_sizes()), device=device)
    return BlockFrame(block, max_batch=size)


def test_concat_stream_input_len(block_frame, concat_frame):
    assert len(block_frame) == len(concat_frame)


def test_concat_stream_next(block_frame, concat_frame):
    stream = ZipExec(iter(block_frame), iter(concat_frame))

    for result, lhs, rhs in zip(stream, block_frame, concat_frame):
        concat = TensorDict({**lhs.data, **rhs.data}, device=result.data.device)
        assert (result.data == concat).all()


@pytest.fixture(scope="module")
def joinable_frame(size, device):
    block = fake.unionable_block_ok(size=size, device=device)
    return BlockFrame(block, max_batch=size)


def test_join_stream_input_len(block_frame, joinable_frame):
    assert len(block_frame) * len(joinable_frame)


def test_join_stream_next(block_frame, joinable_frame):
    stream = MatrixJoinExec(iter(block_frame), joinable_frame, on="i1d")
    for idx, result in enumerate(stream):
        left_idx, right_idx = divmod(idx, len(joinable_frame))

        left = block_frame[left_idx]
        right = joinable_frame[right_idx]

        assert "i1d" in result.keys()
        assert "i1d" in left.keys()
        assert "i1d" in right.keys()

        unique_left = Counter(left["i1d"].tolist())
        unique_right = Counter(right["i1d"].tolist())
        unique_result = Counter(result["i1d"].tolist())

        assert unique_result.keys() == unique_left.keys() & unique_right.keys()

        for key in unique_result.keys():
            assert unique_result[key] == unique_left[key] * unique_right[key]
