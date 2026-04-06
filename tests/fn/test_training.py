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
def trainable_param():
    return torch.randn(7, 3).requires_grad_()


@pytest.fixture
def input(trainable_param: torch.Tensor):
    return defer(trainable_param)


@pytest.fixture
def target_param(trainable_param: torch.Tensor):
    return torch.randn_like(trainable_param)


@pytest.fixture
def target(target_param: torch.Tensor):
    return defer(target_param)


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


def test_backward_fn(
    loss_fn: LossFn,
    input: TensorFn,
    target: TensorFn,
    trainable_param: torch.Tensor,
):
    loss_fn.backward()
    assert input.grad is not None
    assert target.grad is None
    assert (input.grad == trainable_param.grad).all()


def test_optim_zero_grad(
    optimizer: OptimFn,
    loss_fn: LossFn,
    input: TensorFn,
    target: TensorFn,
    trainable_param: torch.Tensor,
):
    loss_fn.backward()
    optimizer.zero_grad()
    assert input.grad is target.grad is None


def test_optim_step(
    optimizer: OptimFn,
    loss_fn: LossFn,
    trainable_param: torch.Tensor,
    target_param: torch.Tensor,
):
    original = trainable_param.clone()
    optimizer.zero_grad()
    assert trainable_param.grad is None
    loss_fn.backward()
    assert trainable_param.grad is not None
    optimizer.step()

    # Test if optimization step did happen.
    updated = trainable_param != original
    no_update_needed = trainable_param == target_param
    assert (updated | no_update_needed).all()
