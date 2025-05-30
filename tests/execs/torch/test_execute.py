# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest


def test_next_of_execute(tracer):
    out = next(tracer.exe)
    assert len(out) == 4

    out = next(tracer.exe)
    assert len(out) == 4

    # Because batch size is 4 and total is 11.
    out = next(tracer.exe)
    assert len(out) == 3

    with pytest.raises(StopIteration):
        next(tracer.exe)
