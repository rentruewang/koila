# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch

from aioway import meta


def _cpus():
    yield "cpu"
    yield torch.device("cpu")


@pytest.fixture(params=_cpus())
def cpu(request: pytest.FixtureRequest):
    return meta.Device.parse(request.param)


def test_eq(cpu: meta.Device):
    assert cpu == "cpu"
    assert cpu == meta.Device.parse("cpu")


def test_no_fail(cpu: meta.Device):
    assert cpu != object()
