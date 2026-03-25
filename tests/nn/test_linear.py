# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensorMode

from aioway.attrs import Attr
from aioway.nn import Linear


@pytest.fixture(autouse=True)
def fake_mode():
    with FakeTensorMode():
        yield


@pytest.fixture
def linear():
    return Linear(
        in_features=3,
        out_features=5,
        bias=True,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


@pytest.fixture
def linear_input():
    return torch.randn(7, 3)


@pytest.fixture
def linear_attr(linear_input: Tensor):
    return Attr.from_tensor(linear_input)


def test_linear_forward(linear: Linear, linear_input: Tensor):
    result = linear.forward(linear_input)
    assert isinstance(result, Tensor)
    assert result.shape == (7, 5)


def test_linear_preview(linear: Linear, linear_attr: Attr):
    result = linear.preview(linear_attr)
    assert isinstance(result, Attr)
    assert result.shape == (7, 5)
