# Copyright (c) AIoWay Authors - All Rights Reserved


def test_block_frame_len(block_frame, batch_size, data_size):
    assert len(block_frame) == data_size // batch_size


def test_frame_iter_is_clone(block_frame, make_executor):
    for left, right in zip(
        make_executor(block_frame.op.thunk()),
        make_executor(block_frame.op.thunk()),
    ):
        assert (left == right).all()
        assert left is not right
