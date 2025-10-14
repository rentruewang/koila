# Copyright (c) AIoWay Authors - All Rights Reserved


from aioway.schedulers import Scheduler
from aioway.tables import Table


def test_block_frame_len(block_frame: Table, batch_size: int, data_size: int):
    assert len(block_frame) == data_size // batch_size


def test_frame_iter_is_clone(block_frame: Table, scheduler: Scheduler):
    for left, right in zip(
        scheduler(block_frame.plan()),
        scheduler(block_frame.plan()),
    ):
        assert (left == right).all()
        assert left is not right
