# Copyright (c) AIoWay Authors - All Rights Reserved

import random

from tests import fake


def test_block_frame_len(block_frame):
    assert len(block_frame) == max(fake.tensordict_sizes())


def test_block_frame_getitem(block_frame):
    idx = random.randrange(len(block_frame))
    assert isinstance(idx, int)

    assert (block_frame[idx] == block_frame.td[idx]).all()
    assert not block_frame[idx].batch_dims


def test_frame_iter_is_clone(block_frame_op, make_executor):
    for left, right in zip(
        make_executor(block_frame_op.thunk()),
        make_executor(block_frame_op.thunk()),
    ):
        assert (left.data == right.data).all()
        assert left is not right


def test_iterator_eq(block_frame_op, another_block_frame_op, make_executor):
    for left, right in zip(
        make_executor(block_frame_op.thunk()),
        make_executor(another_block_frame_op.thunk()),
    ):
        assert (left.data == right.data).all()
