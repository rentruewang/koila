# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn

from aioway import meta, modules


@pytest.fixture
def linear():
    return modules.Module(nn.Linear, in_features=3, out_features=5, bias=True)


@pytest.fixture
def identity():
    return modules.Module(nn.Identity)


@pytest.fixture
def linear_input():
    return torch.randn(7, 3)


@pytest.fixture
def linear_attr(linear_input: torch.Tensor):
    return meta.Attr.from_tensor(linear_input)


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
    return modules.Module(
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
    return modules.Module(nn.Embedding, num_embeddings=3, embedding_dim=5)


def _emb_inputs():
    yield torch.randint(0, 3, size=[11, 7])
    yield torch.randint(0, 3, size=[11, 3])


@pytest.fixture(params=_emb_inputs())
def emb_input(request: pytest.FixtureRequest):
    return request.param


def test_linear(linear: modules.Module, linear_input: torch.Tensor):
    result = linear.forward(linear_input)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (7, 5)


def test_linear_preview(linear: modules.Module, linear_attr: meta.Attr):
    result = linear.preview(linear_attr)
    assert isinstance(result, meta.Attr)
    assert result.shape == (7, 5)


def test_identity(identity: modules.Module, linear_input: torch.Tensor):
    result = identity.forward(linear_input)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (7, 3)


def test_identity_preview(identity: modules.Module, linear_attr: meta.Attr):
    result = identity.preview(linear_attr)
    assert isinstance(result, meta.Attr)
    assert result.shape == (7, 3)


def test_conv2d_forward(conv2d: modules.Module, conv2d_input: torch.Tensor):
    ours = conv2d.forward(conv2d_input)
    theirs = conv2d.real_module(conv2d_input)

    assert ours.shape == theirs.shape


def test_conv2d_preview(conv2d: modules.Module, conv2d_input: torch.Tensor):
    ours = conv2d.preview(meta.Attr.from_tensor(conv2d_input))
    theirs = conv2d.real_module(conv2d_input)

    assert ours.shape == theirs.shape


def test_emb_forward(emb_input: torch.Tensor, emb: modules.Module):
    assert emb.forward(emb_input).shape == emb.real_module(emb_input).shape
    assert emb.forward(emb_input).dtype == emb.real_module(emb_input).dtype


def test_emb_preview(emb_input: torch.Tensor, emb: modules.Module):
    preview = emb.preview(meta.Attr.from_tensor(emb_input))
    real = emb.real_module(emb_input)
    assert preview.shape == real.shape
    assert preview.dtype == real.dtype
