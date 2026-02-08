# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway import attrs
from aioway.attrs import Device


@pytest.fixture
def cpu():
    return attrs.device("cpu")


def test_eq(cpu: Device):
    assert cpu == "cpu"
    assert cpu == attrs.device("cpu")
