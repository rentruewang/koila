# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
from collections.abc import Callable

import pytest

from aioway.calls import CallRewriteMgr


@dcls.dataclass
class CountingFactory:
    """Factory that creates a CountingProc."""

    count: int = 0

    def __call__(self, func: Callable[[float], float]) -> Callable[[float], float]:
        self.count += 1

        def f(x: float) -> float:
            return func(x) + 1

        return f


@pytest.fixture(scope="function")
def static():
    return CountingFactory()


@pytest.fixture(scope="function")
def dynamic():
    return CountingFactory()


@pytest.fixture(scope="function")
def rewrite_manager(static, dynamic):
    """Create a ProcLifetimeManager with static and dynamic processors."""
    return CallRewriteMgr(static=static, dynamic=dynamic)


def test_rewrite_manager_call(rewrite_manager, static, dynamic):
    assert static.count == 0
    assert dynamic.count == 0

    @rewrite_manager
    def func(x):
        return x + 1

    for i in range(10):
        assert static.count == min(i, 1)
        assert dynamic.count == i

        out = func(i + 1)
        assert out == i + 4
