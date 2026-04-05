# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from aioway.fn import LossFn, TensorFn, defer
from aioway.schemas import Shape


def _loss_fns():
    yield nn.L1Loss()
    yield F.l1_loss

    yield nn.MSELoss()
    yield F.mse_loss


@pytest.fixture
def input():
    return defer(torch.randn(7, 3).requires_grad_())


@pytest.fixture
def target():
    return defer(torch.randn(7, 3))


@pytest.fixture(params=_loss_fns())
def loss_fn(request: pytest.FixtureRequest, input: TensorFn, target: TensorFn):
    return LossFn(loss=request.param, input=input, target=target)


def test_loss_fn(loss_fn: LossFn):
    assert isinstance(loss_fn.shape, Shape)
    assert loss_fn.shape.numel() == 1


def test_backward_fn(loss_fn: LossFn, input: TensorFn, target: TensorFn):
    loss_fn.backward()
    assert input.grad is not None
    assert target.grad is None
