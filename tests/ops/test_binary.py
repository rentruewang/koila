# Copyright (c) AIoWay Authors - All Rights Reserved

from collections import Counter

import tensordict
from tensordict import TensorDict

from aioway.plans import MatchPlan, Plan
from aioway.schedulers import Scheduler
from aioway.tables import Table


def test_zip_input_len(block_frame, concat_frame):
    assert len(block_frame) == len(concat_frame)


def test_zip(
    zipper: Plan, block_frame: Table, concat_frame: Table, scheduler: Scheduler
):
    stream = zipper(block_frame.plan(), concat_frame.plan())

    for result, lhs, rhs in zip(
        scheduler(stream),
        scheduler(block_frame.plan()),
        scheduler(concat_frame.plan()),
    ):
        concat = TensorDict({**lhs, **rhs}, device=result.device)
        assert (result == concat).all()


def test_join_input_len(block_frame: Table, joinable_frame: Table):
    assert len(block_frame) * len(joinable_frame)


def test_match_is_reduction(
    matcher: MatchPlan, block_frame: Table, joinable_frame: Table, scheduler: Scheduler
):
    stream = matcher(block_frame.plan(), joinable_frame.plan())
    block_frame_block = tensordict.cat(list(block_frame))
    joinable_frame_block = tensordict.cat(list(joinable_frame))

    # Performing the join here.
    results: list[TensorDict] = list(scheduler(stream))
    answer_items = tensordict.cat(results)["i1d"]

    # Do it at once.
    ground_truth = matcher.join(block_frame_block, joinable_frame_block)

    answer_count = Counter(answer_items.tolist())
    truth_count = Counter(ground_truth["i1d"].tolist())

    assert answer_count == truth_count


def _left_match_right(
    matcher: MatchPlan, left_frame: Table, right_frame: Table, scheduler: Scheduler
):
    stream = matcher(left_frame.plan(), right_frame.plan())
    block_frame_block = tensordict.cat(list(left_frame))
    joinable_frame_block = tensordict.cat(list(right_frame))

    # Performing the join here.
    results: list[TensorDict] = list(scheduler(stream))
    answer_items = tensordict.cat(results)["i1d"]

    answer_count = Counter(answer_items.tolist())

    left_count = Counter(block_frame_block["i1d"].tolist())
    right_count = Counter(joinable_frame_block["i1d"].tolist())

    # Functionally correct join.
    assert left_count.keys() == {*block_frame_block["i1d"].tolist()}
    assert right_count.keys() == {*joinable_frame_block["i1d"].tolist()}
    assert answer_count.keys() == left_count.keys() & right_count.keys()

    for key in answer_count.keys():
        assert answer_count[key] == left_count[key] * right_count[key]


def test_match_functionally_correct(
    matcher, block_frame, joinable_frame, scheduler: Scheduler
):
    _left_match_right(matcher, block_frame, joinable_frame, scheduler)


def test_duplicate_computation(matcher, block_frame, scheduler: Scheduler):
    _left_match_right(matcher, block_frame, block_frame, scheduler)
