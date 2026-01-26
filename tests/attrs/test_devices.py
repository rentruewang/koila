# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway.attrs import Device


@pytest.fixture
def cpu():
    return Device("cpu")


def test_eq(cpu: Device):
    assert cpu == "cpu"
    assert cpu == Device("cpu")
