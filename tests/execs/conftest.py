# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway.execs import ExecInit


@pytest.fixture(scope="module")
def exec_init_reg():
    "The registry for ``ExecInit``."

    return ExecInit.preset()
