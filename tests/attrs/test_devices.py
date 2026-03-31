# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch

from aioway import tensors


def _cpus():
    yield "cpu"
    yield torch.device("cpu")


@pytest.fixture(params=_cpus())
def cpu(request: pytest.FixtureRequest):
    return tensors.Device.parse(request.param)


def test_eq(cpu: tensors.Device):
    assert cpu == "cpu"
    assert cpu == tensors.Device.parse("cpu")


def test_no_fail(cpu: tensors.Device):
    assert cpu != object()
