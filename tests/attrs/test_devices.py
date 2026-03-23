# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from pytest import FixtureRequest

from aioway import attrs


def _cpus():
    yield "cpu"
    yield torch.device("cpu")


@pytest.fixture(params=_cpus())
def cpu(request: FixtureRequest):
    return attrs.device(request.param)


def test_eq(cpu):
    assert cpu == "cpu"
    assert cpu == attrs.device("cpu")


def test_no_fail(cpu):
    assert cpu != object()
