# Copyright (c) AIoWay Authors - All Rights Reserved

from collections import Counter

import pytest
import tensordict
from tensordict import TensorDict

from aioway.blocks import Block
from aioway.ops import MatchOp, ZipOp


@pytest.fixture
def match_op():
    return MatchOp(key="i1d")


def combine_results(results: list[Block], /) -> TensorDict:
    data = [result.data for result in results]
    return tensordict.cat(data)


def test_zip_input_len(block_frame, concat_frame):
    assert len(block_frame) == len(concat_frame)


def test_zip(block_frame_op, concat_frame_op):
    stream = ZipOp().thunk(block_frame_op.thunk(), concat_frame_op.thunk())

    for result, lhs, rhs in zip(
        stream, block_frame_op.thunk(), concat_frame_op.thunk()
    ):
        concat = TensorDict({**lhs.data, **rhs.data}, device=result.data.device)
        assert (result.data == concat).all()


def test_join_input_len(block_frame, joinable_frame):
    assert len(block_frame) * len(joinable_frame)


def test_match_is_reduction(match_op, block_frame_op, joinable_frame_op):
    stream = match_op.thunk(block_frame_op.thunk(), joinable_frame_op.thunk())
    block_frame_block = block_frame_op.dataset.block
    joinable_frame_block = joinable_frame_op.dataset.block

    # Performing the join here.
    results: list[Block] = list(stream)
    answer_items = combine_results(results)["i1d"]

    # Do it at once.
    ground_truth = match_op.join(block_frame_block, joinable_frame_block)

    answer_count = Counter(answer_items.tolist())
    truth_count = Counter(ground_truth.data["i1d"].tolist())

    assert answer_count == truth_count


def test_match_functionally_correct(match_op, block_frame_op, joinable_frame_op):
    stream = match_op.thunk(block_frame_op.thunk(), joinable_frame_op.thunk())
    block_frame_block = block_frame_op.dataset.block
    joinable_frame_block = joinable_frame_op.dataset.block

    # Performing the join here.
    results: list[Block] = list(stream)
    answer_items = combine_results(results)["i1d"]

    answer_count = Counter(answer_items.tolist())

    left_count = Counter(block_frame_block.data["i1d"].tolist())
    right_count = Counter(joinable_frame_block.data["i1d"].tolist())

    # Functionally correct join.
    assert left_count.keys() == {*block_frame_block.data["i1d"].tolist()}
    assert right_count.keys() == {*joinable_frame_block.data["i1d"].tolist()}
    assert answer_count.keys() == left_count.keys() & right_count.keys()

    for key in answer_count.keys():
        assert answer_count[key] == left_count[key] * right_count[key]
