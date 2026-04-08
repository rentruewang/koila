# Copyright (c) AIoWay Authors - All Rights Reserved

import operator
from collections import abc as cabc

import pytest
import torch

from aioway.fn import TensorFn
from aioway.schemas import attr


@pytest.fixture
def left():
    return torch.randn(3, 5)


@pytest.fixture
def right():
    return torch.randn(1, 5)


@pytest.fixture
def left_fn(left: torch.Tensor):
    return TensorFn.from_tensor(left)


@pytest.fixture
def right_fn(right: torch.Tensor):
    return TensorFn.from_tensor(right)


@pytest.fixture
def index():
    return torch.randint(0, 3, [2, 7])


@pytest.fixture
def index_fn(index: torch.Tensor):
    return TensorFn.from_tensor(index)


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


def test_left_normal(left_fn: TensorFn):
    assert isinstance(left_fn.forward(), torch.Tensor)


def test_matmul(left_fn: TensorFn):
    out = left_fn @ torch.randn(5, 7)
    assert out.shape == [3, 7]
    assert out.do().shape == out.shape


def test_matmul_fail(left_fn: TensorFn):
    with pytest.raises(ValueError):
        left_fn @ torch.randn(11, 13)


def test_left_attr(left_fn: TensorFn):
    tensor = left_fn.preview()
    assert isinstance(tensor, torch.Tensor)
    a = attr(tensor)
    assert a.shape == [3, 5]
    assert a.device == "cpu"
    assert a.dtype == "float"


def test_binary_ufunc(
    left_fn: TensorFn,
    right_fn: TensorFn,
    binop: cabc.Callable[[TensorFn, TensorFn], TensorFn],
):
    result = binop(left_fn, right_fn)

    assert isinstance(result, TensorFn)
    assert isinstance(result.forward(), torch.Tensor)


def test_getitem(left_fn: TensorFn, index_fn: TensorFn):
    result = left_fn[index_fn]
    assert isinstance(result, TensorFn)
    assert result.attr.shape == [2, 7, 5]
