# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway.rels import Op0, Op1, Op2, Thunk, Thunk0, Thunk1, Thunk2


def test_block_frame_op(block_frame):
    match (thunk := block_frame.op.thunk()):
        case Thunk0(op=op):
            assert isinstance(op, Op0)
        case _:
            raise RuntimeError

    assert thunk.ARGC == len(list(thunk.inputs()))


def test_unpack_match_op(match_op, block_frame):
    thunk = match_op.thunk(block_frame.op.thunk(), block_frame.op.thunk())
    match thunk:
        case Thunk2(op=op, left=left, right=right):
            assert isinstance(op, Op2)
            assert isinstance(left, Thunk)
            assert isinstance(right, Thunk)
        case _:
            raise RuntimeError
    assert thunk.ARGC == len(list(thunk.inputs()))


def test_unpack_filter_op(filter_op, block_frame):
    thunk = filter_op.thunk(block_frame.op.thunk())
    match thunk:
        case Thunk1(op=op, input=input):
            assert isinstance(op, Op1)
            assert isinstance(input, Thunk)
        case _:
            raise RuntimeError
    assert thunk.ARGC == len(list(thunk.inputs()))
