# Copyright (c) AIoWay Authors - All Rights Reserved

import itertools

import pytest

from aioway.ops import BlockIter, RepeatOp


@pytest.mark.parametrize("times", [1, 2, 4])
def test_repeat_op(block_frame_op, make_executor, times):
    repeat_op = RepeatOp(times=times)

    def consume_batch[T](iterable: BlockIter, times: int):
        it = iter(iterable)
        while True:
            # Call ``next`` ``times`` times.
            if not (result := [blk for _, blk in zip(range(times), it)]):
                return
            yield result

    for repeated, original in itertools.zip_longest(
        consume_batch(
            make_executor(repeat_op.thunk(block_frame_op.thunk())),
            times=times,
        ),
        make_executor(block_frame_op.thunk()),
    ):
        for r in repeated:
            assert (r.data == original.data).all()
