# Copyright (c) AIoWay Authors - All Rights Reserved

import operator
from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from aioway import fake
from aioway._previews import Attr
from aioway.fn import TensorFn


@pytest.fixture
def left():
    return torch.randn(3, 5)


@pytest.fixture
def right():
    return torch.randn(1, 5)


@pytest.fixture
def left_fn(left: Tensor):
    return TensorFn.from_tensor(left)


@pytest.fixture
def right_fn(right: Tensor):
    return TensorFn.from_tensor(right)


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
    with fake.enable() as f:
        yield f


def test_left_normal(left_fn: TensorFn):
    assert isinstance(left_fn.do(), Tensor)


def test_left_fake_forward(left_fn: TensorFn):
    with fake.enable():
        assert fake.is_fake_tensor(left_fn.forward())


def test_left_fake_do(left_fn: TensorFn):
    with fake.enable():
        assert fake.is_fake_tensor(left_fn.do())


def test_left_attr(left_fn: TensorFn):
    attr = left_fn.attr()
    assert isinstance(attr, Attr)
    assert attr.shape == [3, 5]
    assert attr.device == "cpu"
    assert attr.dtype == "float"


def test_binary_ufunc(
    left_fn: TensorFn,
    right_fn: TensorFn,
    binop: Callable[[TensorFn, TensorFn], TensorFn],
):
    result = binop(left_fn, right_fn)

    assert isinstance(result, TensorFn)
    assert isinstance(result.do(), Tensor)
