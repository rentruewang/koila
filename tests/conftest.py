# Copyright (c) AIoWay Authors - All Rights Reserved

import random

import pytest
import torch
from numpy import random as npr
from pytest import FixtureRequest
from rich import traceback

from . import fake


@pytest.fixture(autouse=True, scope="session")
def enable_traceback():
    """
    Enable rich traceback for all tests.
    """
    traceback.install(show_locals=True, word_wrap=True)


@pytest.fixture(autouse=True, scope="session")
def seed():
    seed = 42
    random.seed(seed)
    npr.seed(seed)
    torch.manual_seed(seed)
    return seed


@pytest.fixture(params=fake.cpu_and_maybe_cuda())
def device(request: FixtureRequest) -> str:
    return request.param


@pytest.fixture
def data_size() -> int:
    return max(fake.batch_sizes())


@pytest.fixture(params=fake.batch_sizes())
def batch_size(request: FixtureRequest) -> int:
    return request.param
