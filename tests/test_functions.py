import torch

import koila
from koila import LazyTensor
import math
from .common import ArgsKwargs, Condition


def test_scalar_add_op() -> None:
    Condition(
        lambda a, b, c: koila.run((a + b)).item() == c,
        [
            ArgsKwargs(LazyTensor(torch.tensor(1)), LazyTensor(torch.tensor(2)), 3),
            ArgsKwargs(torch.tensor(1), LazyTensor(torch.tensor(2)), 3),
            ArgsKwargs(LazyTensor(torch.tensor(1)), torch.tensor(2), 3),
        ],
    ).check()


def test_scalar_add_method() -> None:
    Condition(
        lambda a, b, c: koila.run(a.add(b)).item() == c,
        [
            ArgsKwargs(LazyTensor(torch.tensor(4)), LazyTensor(torch.tensor(3)), 7),
            ArgsKwargs(torch.tensor(4), LazyTensor(torch.tensor(3)), 7),
            ArgsKwargs(LazyTensor(torch.tensor(4)), torch.tensor(3), 7),
        ],
    ).check()


def test_scalar_add_function() -> None:
    Condition(
        lambda a, b, c: koila.run(torch.add(a, b)).item() == c,
        [
            ArgsKwargs(LazyTensor(torch.tensor(8)), LazyTensor(torch.tensor(4)), 12),
            ArgsKwargs(torch.tensor(8), LazyTensor(torch.tensor(4)), 12),
            ArgsKwargs(LazyTensor(torch.tensor(8)), torch.tensor(4), 12),
        ],
    ).check()


def test_scalar_sub_op() -> None:
    Condition(
        lambda a, b, c: koila.run((a - b)).item() == c,
        [
            ArgsKwargs(LazyTensor(torch.tensor(1)), LazyTensor(torch.tensor(2)), -1),
            ArgsKwargs(torch.tensor(1), LazyTensor(torch.tensor(2)), -1),
            ArgsKwargs(LazyTensor(torch.tensor(1)), torch.tensor(2), -1),
        ],
    ).check()


def test_scalar_sub_method() -> None:
    Condition(
        lambda a, b, c: koila.run(a.sub(b)).item() == c,
        [
            ArgsKwargs(LazyTensor(torch.tensor(4)), LazyTensor(torch.tensor(3)), 1),
            ArgsKwargs(torch.tensor(4), LazyTensor(torch.tensor(3)), 1),
            ArgsKwargs(LazyTensor(torch.tensor(4)), torch.tensor(3), 1),
        ],
    ).check()


def test_scalar_sub_function() -> None:
    Condition(
        lambda a, b, c: koila.run(torch.sub(a, b)).item() == c,
        [
            ArgsKwargs(LazyTensor(torch.tensor(8)), LazyTensor(torch.tensor(4)), 4),
            ArgsKwargs(torch.tensor(8), LazyTensor(torch.tensor(4)), 4),
            ArgsKwargs(LazyTensor(torch.tensor(8)), torch.tensor(4), 4),
        ],
    ).check()


def test_scalar_mul_op() -> None:
    Condition(
        lambda a, b, c: koila.run((a * b)).item() == c,
        [
            ArgsKwargs(LazyTensor(torch.tensor(1)), LazyTensor(torch.tensor(2)), 2),
            ArgsKwargs(torch.tensor(1), LazyTensor(torch.tensor(2)), 2),
            ArgsKwargs(LazyTensor(torch.tensor(1)), torch.tensor(2), 2),
        ],
    ).check()


def test_scalar_mul_method() -> None:
    Condition(
        lambda a, b, c: koila.run(a.mul(b)).item() == c,
        [
            ArgsKwargs(LazyTensor(torch.tensor(4)), LazyTensor(torch.tensor(3)), 12),
            ArgsKwargs(torch.tensor(4), LazyTensor(torch.tensor(3)), 12),
            ArgsKwargs(LazyTensor(torch.tensor(4)), torch.tensor(3), 12),
        ],
    ).check()


def test_scalar_mul_function() -> None:
    Condition(
        lambda a, b, c: koila.run(torch.mul(a, b)).item() == c,
        [
            ArgsKwargs(LazyTensor(torch.tensor(8)), LazyTensor(torch.tensor(4)), 32),
            ArgsKwargs(torch.tensor(8), LazyTensor(torch.tensor(4)), 32),
            ArgsKwargs(LazyTensor(torch.tensor(8)), torch.tensor(4), 32),
        ],
    ).check()


def test_scalar_floordiv_op() -> None:
    Condition(
        lambda a, b, c: koila.run((a // b)).item() == c,
        [
            ArgsKwargs(LazyTensor(torch.tensor(1)), LazyTensor(torch.tensor(2)), 0),
            ArgsKwargs(torch.tensor(1), LazyTensor(torch.tensor(2)), 0),
            ArgsKwargs(LazyTensor(torch.tensor(1)), torch.tensor(2), 0),
        ],
    ).check()


def test_scalar_floordiv_method() -> None:
    Condition(
        lambda a, b, c: koila.run(a.div(b, rounding_mode="trunc")).item() == c,
        [
            ArgsKwargs(LazyTensor(torch.tensor(4)), LazyTensor(torch.tensor(3)), 1),
            ArgsKwargs(torch.tensor(4), LazyTensor(torch.tensor(3)), 1),
            ArgsKwargs(LazyTensor(torch.tensor(4)), torch.tensor(3), 1),
        ],
    ).check()


def test_scalar_floordiv_function() -> None:
    Condition(
        lambda a, b, c: koila.run(torch.div(a, b, rounding_mode="trunc")).item() == c,
        [
            ArgsKwargs(LazyTensor(torch.tensor(9)), LazyTensor(torch.tensor(4)), 2),
            ArgsKwargs(torch.tensor(9), LazyTensor(torch.tensor(4)), 2),
            ArgsKwargs(LazyTensor(torch.tensor(9)), torch.tensor(4), 2),
        ],
    ).check()


def test_scalar_truediv_op() -> None:
    Condition(
        lambda a, b, c: math.isclose(koila.run((a / b)).item(), c, abs_tol=1e-7),
        [
            ArgsKwargs(LazyTensor(torch.tensor(1)), LazyTensor(torch.tensor(2)), 1 / 2),
            ArgsKwargs(torch.tensor(1), LazyTensor(torch.tensor(2)), 1 / 2),
            ArgsKwargs(LazyTensor(torch.tensor(1)), torch.tensor(2), 1 / 2),
        ],
    ).check()


def test_scalar_truediv_method() -> None:
    Condition(
        lambda a, b, c: math.isclose(koila.run(a.div(b)).item(), c, abs_tol=1e-7),
        [
            ArgsKwargs(LazyTensor(torch.tensor(4)), LazyTensor(torch.tensor(3)), 4 / 3),
            ArgsKwargs(torch.tensor(4), LazyTensor(torch.tensor(3)), 4 / 3),
            ArgsKwargs(LazyTensor(torch.tensor(4)), torch.tensor(3), 4 / 3),
        ],
    ).check()


def test_scalar_truediv_function() -> None:
    Condition(
        lambda a, b, c: math.isclose(
            koila.run(torch.div(a, b)).item(), c, abs_tol=1e-7
        ),
        [
            ArgsKwargs(LazyTensor(torch.tensor(9)), LazyTensor(torch.tensor(4)), 9 / 4),
            ArgsKwargs(torch.tensor(9), LazyTensor(torch.tensor(4)), 9 / 4),
            ArgsKwargs(LazyTensor(torch.tensor(9)), torch.tensor(4), 9 / 4),
        ],
    ).check()
