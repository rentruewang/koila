# Copyright (c) AIoWay Authors - All Rights Reserved


import itertools
from collections.abc import Generator

import pytest
from tensordict import TensorDict

from aioway.plans import BatchIter, Plan
from aioway.schedulers import Scheduler
from aioway.tables import Table


@pytest.mark.parametrize("times", [1, 2, 4])
def test_repeat(repeater: Plan, block_frame: Table, scheduler: Scheduler, times: int):

    def chunk_batch(iterable: BatchIter, times: int) -> Generator[list[TensorDict]]:
        """
        Sample for each ``times`` times.
        """

        # Cast to generator because ``SharedExec`` can be iterated multiple times.
        it = (item for item in iterable)
        while True:
            # Call ``next`` ``times`` times.
            if not (result := [blk for _, blk in zip(range(times), it)]):
                return
            yield result

    # First repeat ``times`` times.
    repeat_iter = scheduler(repeater(block_frame.plan()))
    block_frame_iter = scheduler(block_frame.plan())
    for repeated, original in itertools.zip_longest(
        chunk_batch(repeat_iter, times=times), block_frame_iter
    ):
        assert repeated is not None, "Should be of equal length."
        assert original is not None, "Should be of equal length."
        assert isinstance(repeated, list)
        assert all(isinstance(r, TensorDict) for r in repeated)
        for r in repeated:
            assert (r == original).all()
