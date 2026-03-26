# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
from torch._subclasses.fake_tensor import FakeTensorMode


@pytest.fixture(autouse=True)
def fake_mode():
    "Enable `FakeTensorMode` for the entire module, s.t. testing runs faster."

    with FakeTensorMode():
        yield
