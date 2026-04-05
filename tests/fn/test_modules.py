# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn

from aioway import schemas
from aioway.fn import modules


@pytest.fixture
def linear():
    return modules.FakableModule(nn.Linear, in_features=3, out_features=5, bias=True)


@pytest.fixture
def identity():
    return modules.FakableModule(nn.Identity)


@pytest.fixture
def linear_input():
    return torch.randn(7, 3)


@pytest.fixture
def linear_attr(linear_input: torch.Tensor):
    return schemas.Attr.from_tensor(linear_input)


@pytest.fixture
def conv2d_input():
    return torch.randn(3, 5, 7, 11)


@pytest.fixture(params=[1, 2, 3])
def dilation(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=[0, 1, 2, 3])
def padding(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def stride(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def kernel_size(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture
def conv2d(dilation: int, padding: int, stride: int, kernel_size: int):
    return modules.FakableModule(
        nn.Conv2d,
        in_channels=5,
        out_channels=13,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )


@pytest.fixture
def emb():
    return modules.FakableModule(nn.Embedding, num_embeddings=3, embedding_dim=5)


def _emb_inputs():
    yield torch.randint(0, 3, size=[11, 7])
    yield torch.randint(0, 3, size=[11, 3])


@pytest.fixture(params=_emb_inputs())
def emb_input(request: pytest.FixtureRequest):
    return request.param


def test_linear(linear: modules.FakableModule, linear_input: torch.Tensor):
    result = linear(linear_input)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (7, 5)


def test_identity(identity: modules.FakableModule, linear_input: torch.Tensor):
    result = identity(linear_input)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (7, 3)


def test_conv2d_forward(conv2d: modules.FakableModule, conv2d_input: torch.Tensor):
    ours = conv2d(conv2d_input)
    theirs = conv2d.real(conv2d_input)

    assert ours.shape == theirs.shape


def test_emb_forward(emb_input: torch.Tensor, emb: modules.FakableModule):
    assert emb(emb_input).shape == emb.real(emb_input).shape
    assert emb(emb_input).dtype == emb.real(emb_input).dtype
