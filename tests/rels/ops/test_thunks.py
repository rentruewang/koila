# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway.rels import Plan0, Plan1, Plan2, Thunk, Thunk0, Thunk1, Thunk2


def test_block_frame_op(block_frame):
    match (thunk := block_frame.op.thunk()):
        case Thunk0(plan=plan):
            assert isinstance(plan, Plan0)
        case _:
            raise RuntimeError

    assert thunk.ARGC == len(list(thunk.inputs()))


def test_unpack_match_op(match_op, block_frame):
    thunk = match_op.thunk(block_frame.op.thunk(), block_frame.op.thunk())
    match thunk:
        case Thunk2(plan=plan, left=left, right=right):
            assert isinstance(plan, Plan2)
            assert isinstance(left, Thunk)
            assert isinstance(right, Thunk)
        case _:
            raise RuntimeError
    assert thunk.ARGC == len(list(thunk.inputs()))


def test_unpack_filter_op(filter_op, block_frame):
    thunk = filter_op.thunk(block_frame.op.thunk())
    match thunk:
        case Thunk1(plan=plan, input=input):
            assert isinstance(plan, Plan1)
            assert isinstance(input, Thunk)
        case _:
            raise RuntimeError
    assert thunk.ARGC == len(list(thunk.inputs()))
