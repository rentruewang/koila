# Copyright (c) AIoWay Authors - All Rights Reserved

import numpy as np
import pytest
from numpy import random

from aioway import chunks, datasets


def test_table_not_empty(frame: datasets.Frame):
    assert frame
    assert len(frame)


def test_table_idx_arr(frame: datasets.Frame):
    idx = random.randint(low=-len(frame), high=len(frame), size=[len(frame)])

    assert np.all(-len(frame) <= idx)
    assert np.all(idx < len(frame))
    assert idx.shape == (len(frame),)

    out = frame[idx]
    assert isinstance(out, chunks.Chunk)
    assert len(out) == len(idx)


def test_table_idx_slice(frame: datasets.Frame):
    out = frame[-len(frame) : len(frame)]
    assert isinstance(out, chunks.Chunk)
    assert len(out) == len(frame)


def test_table_out_of_bounds(frame: datasets.Frame):
    with pytest.raises(IndexError):
        _ = frame[[-2 * len(frame)]]
