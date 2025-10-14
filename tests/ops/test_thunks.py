# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway.plans import MatchPlan, Plan, Plan0, Plan1, Plan2
from aioway.tables import Table
from aioway.thunks import Thunk, Thunk0, Thunk1, Thunk2


def test_block_frame_op(block_frame: Table):
    match (thunk := block_frame.plan()):
        case Thunk0(plan=plan):
            assert isinstance(plan, Plan0)
        case _:
            raise RuntimeError

    assert thunk.ARGC == len(list(thunk.inputs()))


def test_unpack_match_op(matcher: MatchPlan, block_frame: Table):
    thunk = matcher(block_frame.plan(), block_frame.plan())
    match thunk:
        case Thunk2(plan=plan, left=left, right=right):
            assert isinstance(plan, Plan2)
            assert isinstance(left, Thunk)
            assert isinstance(right, Thunk)
        case _:
            raise RuntimeError
    assert thunk.ARGC == len(list(thunk.inputs()))


def test_unpack_filter_op(filterer: Plan, block_frame: Table):
    thunk = filterer(block_frame.plan())
    match thunk:
        case Thunk1(plan=plan, input=input):
            assert isinstance(plan, Plan1)
            assert isinstance(input, Thunk)
        case _:
            raise RuntimeError
    assert thunk.ARGC == len(list(thunk.inputs()))
