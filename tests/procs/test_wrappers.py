# Copyright (c) AIoWay Authors - All Rights Reserved

import inspect

import numpy as np
import pytest

from aioway.procs import OpaqueProc

from . import utils


@pytest.fixture(params=utils.funcs())
def func(request):
    """Fixture to provide a function for testing."""
    return request.param


@pytest.fixture(params=1.1 ** np.arange(10))
def data(request):
    """Fixture to provide data for testing."""
    return request.param


def test_wrapper_proc_inspect(func):
    """
    Test that WrapperProc preserves the wrapped function's metadata.
    """

    wrapped = OpaqueProc(func)

    assert inspect.signature(wrapped) == inspect.signature(func)


@pytest.mark.parametrize("data", 1.1 ** np.arange(10))
def test_wrapper_proc_exec(func, data):
    """
    Test that WrapperProc executes the wrapping logic defined in _wrap.
    """

    wrapped = OpaqueProc(func)
    result = wrapped(data)
    assert func(data) == pytest.approx(result)
