# Copyright (c) AIoWay Authors - All Rights Reserved

import numpy as np
import pytest
from numpy import random

from aioway.batches import Chunk


def test_table_not_empty(frame):
    assert frame
    assert len(frame)


def test_table_idx_arr(frame):
    idx = random.randint(low=-len(frame), high=len(frame), size=[len(frame)])

    assert np.all(-len(frame) <= idx)
    assert np.all(idx < len(frame))
    assert idx.shape == (len(frame),)

    out = frame[idx]
    assert isinstance(out, Chunk)
    assert len(out) == len(idx)


def test_table_idx_slice(frame):
    out = frame[-len(frame) : len(frame)]
    assert isinstance(out, Chunk)
    assert len(out) == len(frame)


def test_table_out_of_bounds(frame):
    with pytest.raises(IndexError):
        _ = frame[[-2 * len(frame)]]
