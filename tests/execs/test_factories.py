# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway.execs import ExecInit


@pytest.fixture(scope="module")
def exec_factory():
    return ExecInit.preset()


def test_exec_factory_getitem(exec_factory):
    for key in exec_factory:
        result = exec_factory[key]
        assert isinstance(result, ExecInit)
