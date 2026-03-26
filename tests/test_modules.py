# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from pytest import FixtureRequest
from torch import Tensor
from torch.nn import Conv2d, Identity, Linear

from aioway.attrs import Attr
from aioway.modules import Module


@pytest.fixture
def linear():
    return Module(Linear, in_features=3, out_features=5, bias=True)


@pytest.fixture
def identity():
    return Module(Identity)


@pytest.fixture
def linear_input():
    return torch.randn(7, 3)


@pytest.fixture
def linear_attr(linear_input: Tensor):
    return Attr.from_tensor(linear_input)


@pytest.fixture
def conv2d_input():
    return torch.randn(3, 5, 7, 11)


@pytest.fixture(params=[1, 2, 3])
def dilation(request: FixtureRequest):
    return request.param


@pytest.fixture(params=[0, 1, 2, 3])
def padding(request: FixtureRequest):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def stride(request: FixtureRequest):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def kernel_size(request: FixtureRequest):
    return request.param


@pytest.fixture
def conv2d(dilation: int, padding: int, stride: int, kernel_size: int):
    return Module(
        Conv2d,
        in_channels=5,
        out_channels=13,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )


def test_linear(linear: Module, linear_input: Tensor):
    result = linear.forward(linear_input)
    assert isinstance(result, Tensor)
    assert result.shape == (7, 5)


def test_linear_preview(linear: Module, linear_attr: Attr):
    result = linear.preview(linear_attr)
    assert isinstance(result, Attr)
    assert result.shape == (7, 5)


def test_identity(identity: Module, linear_input: Tensor):
    result = identity.forward(linear_input)
    assert isinstance(result, Tensor)
    assert result.shape == (7, 3)


def test_identity_preview(identity: Module, linear_attr: Attr):
    result = identity.preview(linear_attr)
    assert isinstance(result, Attr)
    assert result.shape == (7, 3)


def test_conv2d_forward(conv2d: Module, conv2d_input: Tensor):
    ours = conv2d.forward(conv2d_input)
    theirs = conv2d.real_module(conv2d_input)

    assert ours.shape == theirs.shape


def test_conv2d_preview(conv2d: Module, conv2d_input: Tensor):
    ours = conv2d.preview(Attr.from_tensor(conv2d_input))
    theirs = conv2d.real_module(conv2d_input)

    assert ours.shape == theirs.shape
