# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch

from aioway import schemas


def _cpus():
    yield "cpu"
    yield torch.device("cpu")


@pytest.fixture(params=_cpus())
def cpu(request: pytest.FixtureRequest):
    return schemas.Device.parse(request.param)


def test_eq(cpu: schemas.Device):
    assert cpu == "cpu"
    assert cpu == schemas.Device.parse("cpu")


def test_no_fail(cpu: schemas.Device):
    assert cpu != object()
