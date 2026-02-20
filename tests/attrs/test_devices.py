# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway import attrs


@pytest.fixture
def cpu():
    return attrs.device("cpu")


def test_eq(cpu):
    assert cpu == "cpu"
    assert cpu == attrs.device("cpu")
