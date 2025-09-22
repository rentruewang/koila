# Copyright (c) AIoWay Authors - All Rights Reserved

from collections import Counter
from collections.abc import Callable

import pytest
import tensordict
from tensordict import TensorDict

from aioway.execs import Exec
from aioway.io import Frame
from aioway.ops import MatchOp, Thunk, ZipOp


@pytest.fixture
def match_op():
    return MatchOp(key="i1d")


def test_zip_input_len(block_frame, concat_frame):
    assert len(block_frame) == len(concat_frame)


def test_zip(block_frame, concat_frame, make_executor):
    stream = ZipOp().thunk(block_frame.op.thunk(), concat_frame.op.thunk())

    for result, lhs, rhs in zip(
        make_executor(stream),
        make_executor(block_frame.op.thunk()),
        make_executor(concat_frame.op.thunk()),
    ):
        concat = TensorDict({**lhs.data, **rhs.data}, device=result.data.device)
        assert (result.data == concat).all()


def test_join_input_len(block_frame, joinable_frame):
    assert len(block_frame) * len(joinable_frame)


def test_match_is_reduction(match_op, block_frame, joinable_frame, make_executor):
    stream = match_op.thunk(block_frame.op.thunk(), joinable_frame.op.thunk())
    block_frame_block = tensordict.cat(list(block_frame))
    joinable_frame_block = tensordict.cat(list(joinable_frame))

    # Performing the join here.
    results: list[TensorDict] = list(make_executor(stream))
    answer_items = tensordict.cat(results)["i1d"]

    # Do it at once.
    ground_truth = match_op.join(block_frame_block, joinable_frame_block)

    answer_count = Counter(answer_items.tolist())
    truth_count = Counter(ground_truth.data["i1d"].tolist())

    assert answer_count == truth_count


def _left_match_right(
    match_op: MatchOp,
    left_frame: Frame,
    right_frame: Frame,
    make_executor: Callable[[Thunk], Exec],
):
    stream = match_op.thunk(left_frame.op.thunk(), right_frame.op.thunk())
    block_frame_block = tensordict.cat(list(left_frame))
    joinable_frame_block = tensordict.cat(list(right_frame))

    # Performing the join here.
    results: list[TensorDict] = list(make_executor(stream))
    answer_items = tensordict.cat(results)["i1d"]

    answer_count = Counter(answer_items.tolist())

    left_count = Counter(block_frame_block.data["i1d"].tolist())
    right_count = Counter(joinable_frame_block.data["i1d"].tolist())

    # Functionally correct join.
    assert left_count.keys() == {*block_frame_block.data["i1d"].tolist()}
    assert right_count.keys() == {*joinable_frame_block.data["i1d"].tolist()}
    assert answer_count.keys() == left_count.keys() & right_count.keys()

    for key in answer_count.keys():
        assert answer_count[key] == left_count[key] * right_count[key]


def test_match_functionally_correct(
    match_op, block_frame, joinable_frame, make_executor
):
    _left_match_right(match_op, block_frame, joinable_frame, make_executor)


def test_duplicate_computation(match_op, block_frame, make_executor, exec_strat):
    _left_match_right(match_op, block_frame, block_frame, make_executor)
