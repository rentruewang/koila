# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
from collections.abc import Callable

import numpy as np
import pytest

from aioway.calls import MemoCall

from . import utils


@dcls.dataclass
class CallCount:
    func: Callable[[float], float]
    count: int = 0

    def __post_init__(self):
        self.count = 0

    def __repr__(self):
        return str(self.func)

    def __call__(self, x: float) -> float:
        self.count += 1
        return self.func(x)

    def __hash__(self) -> int:
        return hash(self.func)


def repeat_call(func: Callable[[float], float], times: int) -> Callable[[float], None]:
    def function(x: float):
        for _ in range(times):
            func(x)

    return function


@pytest.fixture(params=1.1 ** np.arange(10))
def data(request):
    """Fixture to provide data for testing."""
    return request.param


@pytest.fixture(params=utils.funcs())
def func(request):
    """Fixture to provide a function for testing."""
    return request.param


@pytest.fixture(params=[0, 1, 3, 5, 7])
def times(request):
    """Fixture to provide a number of times for testing."""
    return request.param


@pytest.mark.parametrize(
    "proc_init,golden",
    [
        (lambda x: x, lambda x: x),
        (MemoCall, lambda x: min(x, 1)),
    ],
)
def test_memo_proc_call(func, times, data, proc_init, golden):
    counter = CallCount(func)
    proc = proc_init(counter)

    caller = repeat_call(proc, times)
    caller(data)

    assert counter.count == golden(times)
