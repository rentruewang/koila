# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from pytest import FixtureRequest

from aioway.tensors import Device


def _cpus():
    yield "cpu"
    yield torch.device("cpu")


@pytest.fixture(params=_cpus())
def cpu(request: FixtureRequest):
    return Device.parse(request.param)


def test_eq(cpu: Device):
    assert cpu == "cpu"
    assert cpu == Device.parse("cpu")


def test_no_fail(cpu: Device):
    assert cpu != object()
