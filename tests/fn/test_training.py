# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn, optim
from torch.nn import functional as F

from aioway.fn import LossFn, OptimFn, TensorFn, defer
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


def _optimizer_types():
    yield optim.SGD
    yield optim.AdamW
    yield optim.Adam
    yield optim.RMSprop
    yield optim.NAdam


@pytest.fixture(params=_optimizer_types())
def optim_type(request: pytest.FixtureRequest):
    return request.param


def _lrs():
    yield 0.1
    yield 1e-3
    yield 2e-5


@pytest.fixture(params=_lrs())
def lr(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture
def optimizer(optim_type: type[optim.Optimizer], loss_fn: LossFn, lr: float):
    return OptimFn(optim_cls=optim_type, params=loss_fn.parameters(), lr=lr)


def test_loss_fn(loss_fn: LossFn):
    assert isinstance(loss_fn.shape, Shape)
    assert loss_fn.shape.numel() == 1


def test_backward_fn(loss_fn: LossFn, input: TensorFn, target: TensorFn):
    loss_fn.backward()
    assert input.grad is not None
    assert target.grad is None


def test_optim_zero_grad(
    optimizer: OptimFn, loss_fn: LossFn, input: TensorFn, target: TensorFn
):
    loss_fn.backward()
    optimizer.zero_grad()
    assert input.grad is target.grad is None


def test_optim_step(
    optimizer: OptimFn, loss_fn: LossFn, input: TensorFn, target: TensorFn
):
    original = input.data
    optimizer.zero_grad()
    loss_fn.backward()
    optimizer.step()
    assert torch.all((input.data == original))
