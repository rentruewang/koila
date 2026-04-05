# Copyright (c) AIoWay Authors - All Rights Reserved

import operator
from collections import abc as cabc

import pytest
import torch

from aioway import ctx, fn, schemas


@pytest.fixture
def left():
    return torch.randn(3, 5)


@pytest.fixture
def right():
    return torch.randn(1, 5)


@pytest.fixture
def left_fn(left: torch.Tensor):
    return fn.TensorFn.from_tensor(left)


@pytest.fixture
def right_fn(right: torch.Tensor):
    return fn.TensorFn.from_tensor(right)


@pytest.fixture
def index():
    return torch.randint(0, 3, [2, 7])


@pytest.fixture
def index_fn(index: torch.Tensor):
    return fn.TensorFn.from_tensor(index)


@pytest.fixture(
    params=[
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        operator.floordiv,
        operator.mod,
        operator.pow,
        operator.eq,
        operator.ne,
        operator.gt,
        operator.ge,
        operator.lt,
        operator.le,
    ]
)
def binop(request):
    return request.param


@pytest.fixture
def fake_mode():
    with ctx.fake_mode() as f:
        yield f


def test_left_normal(left_fn: fn.TensorFn):
    assert isinstance(left_fn.forward(), torch.Tensor)


def test_left_attr(left_fn: fn.TensorFn):
    tensor = left_fn.preview()
    assert isinstance(tensor, torch.Tensor)
    attr = schemas.attr(tensor)
    assert attr.shape == [3, 5]
    assert attr.device == "cpu"
    assert attr.dtype == "float"


def test_binary_ufunc(
    left_fn: fn.TensorFn,
    right_fn: fn.TensorFn,
    binop: cabc.Callable[[fn.TensorFn, fn.TensorFn], fn.TensorFn],
):
    result = binop(left_fn, right_fn)

    assert isinstance(result, fn.TensorFn)
    assert isinstance(result.forward(), torch.Tensor)


def test_getitem(left_fn: fn.TensorFn, index_fn: fn.TensorFn):
    result = left_fn[index_fn]
    assert isinstance(result, fn.TensorFn)
    assert result.attr.shape == [2, 7, 5]
