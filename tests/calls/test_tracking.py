# Copyright (c) RenChu Wang - All Rights Reserved

from collections.abc import Callable

import pytest

from aioway.calls import StackCall


@pytest.fixture(scope="function")
def stack():
    return []


def test_stack_with_custom_functions(stack):
    """Test stack with actual functions from utils."""

    def wrap_stack_proc(func: Callable[[], None]):
        return StackCall(func=func, stack=stack)

    @wrap_stack_proc
    def func_1():
        assert len(stack) == 1
        assert stack == [func_1.func]
        return 1 + func_2()

    @wrap_stack_proc
    def func_2():
        assert len(stack) == 2
        assert stack == [func_1.func, func_2.func]
        return 2 + func_3()

    @wrap_stack_proc
    def func_3():
        assert len(stack) == 3
        assert stack == [func_1.func, func_2.func, func_3.func]
        return 3

    result = func_1()
    assert result == 6
