import math

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

import koila
from koila import LazyTensor

from .common import Condition


def scalar_isclose(a: float, b: float) -> bool:
    return math.isclose(a, b, abs_tol=1e-6)


def tensor_allclose(a: Tensor, b: Tensor) -> bool:
    return a.allclose(b)


def array_allclose(a: ndarray, b: ndarray) -> bool:
    return np.allclose(a, b, atol=1e-5)


def test_scalar_positive_op() -> None:
    Condition(
        lambda a, c: scalar_isclose(koila.run(+a).item(), c),
        [[LazyTensor(torch.tensor(-11)), -11]],
    ).check()


def test_scalar_positive_method() -> None:
    Condition(
        lambda a, c: scalar_isclose(koila.run(a.positive()).item(), c),
        [[LazyTensor(torch.tensor(4)), 4]],
    ).check()


def test_scalar_positive_function() -> None:
    Condition(
        lambda a, c: scalar_isclose(koila.run(torch.positive(a)).item(), c),
        [[LazyTensor(torch.tensor(-8)), -8]],
    ).check()


def test_scalar_negative_op() -> None:
    Condition(
        lambda a, c: scalar_isclose(koila.run(-a).item(), c),
        [[LazyTensor(torch.tensor(-13)), 13]],
    ).check()


def test_scalar_negative_method() -> None:
    Condition(
        lambda a, c: scalar_isclose(koila.run(a.neg()).item(), c),
        [[LazyTensor(torch.tensor(2)), -2]],
    ).check()


def test_scalar_negative_function() -> None:
    Condition(
        lambda a, c: scalar_isclose(koila.run(torch.neg(a)).item(), c),
        [[LazyTensor(torch.tensor(-5)), 5]],
    ).check()


def test_scalar_add_op() -> None:
    Condition(
        lambda a, b, c: scalar_isclose(koila.run(a + b).item(), c),
        [
            [LazyTensor(torch.tensor(1)), LazyTensor(torch.tensor(2)), 1 + 2],
            [torch.tensor(1), LazyTensor(torch.tensor(2)), 1 + 2],
            [LazyTensor(torch.tensor(1)), torch.tensor(2), 1 + 2],
        ],
    ).check()


def test_scalar_add_method() -> None:
    Condition(
        lambda a, b, c: scalar_isclose(koila.run(a.add(b)).item(), c),
        [
            [LazyTensor(torch.tensor(4)), LazyTensor(torch.tensor(3)), 4 + 3],
            [torch.tensor(4), LazyTensor(torch.tensor(3)), 4 + 3],
            [LazyTensor(torch.tensor(4)), torch.tensor(3), 4 + 3],
        ],
    ).check()


def test_scalar_add_function() -> None:
    Condition(
        lambda a, b, c: scalar_isclose(koila.run(torch.add(a, b)).item(), c),
        [
            [LazyTensor(torch.tensor(8)), LazyTensor(torch.tensor(4)), 8 + 4],
            [torch.tensor(8), LazyTensor(torch.tensor(4)), 8 + 4],
            [LazyTensor(torch.tensor(8)), torch.tensor(4), 8 + 4],
        ],
    ).check()


def test_scalar_sub_op() -> None:
    Condition(
        lambda a, b, c: scalar_isclose(koila.run(a - b).item(), c),
        [
            [LazyTensor(torch.tensor(1)), LazyTensor(torch.tensor(2)), 1 - 2],
            [torch.tensor(1), LazyTensor(torch.tensor(2)), 1 - 2],
            [LazyTensor(torch.tensor(1)), torch.tensor(2), 1 - 2],
        ],
    ).check()


def test_scalar_sub_method() -> None:
    Condition(
        lambda a, b, c: scalar_isclose(koila.run(a.sub(b)).item(), c),
        [
            [LazyTensor(torch.tensor(4)), LazyTensor(torch.tensor(3)), 4 - 3],
            [torch.tensor(4), LazyTensor(torch.tensor(3)), 4 - 3],
            [LazyTensor(torch.tensor(4)), torch.tensor(3), 4 - 3],
        ],
    ).check()


def test_scalar_sub_function() -> None:
    Condition(
        lambda a, b, c: scalar_isclose(koila.run(torch.sub(a, b)).item(), c),
        [
            [LazyTensor(torch.tensor(8)), LazyTensor(torch.tensor(4)), 8 - 4],
            [torch.tensor(8), LazyTensor(torch.tensor(4)), 8 - 4],
            [LazyTensor(torch.tensor(8)), torch.tensor(4), 8 - 4],
        ],
    ).check()


def test_scalar_mul_op() -> None:
    Condition(
        lambda a, b, c: scalar_isclose(koila.run(a * b).item(), c),
        [
            [LazyTensor(torch.tensor(0.5)), LazyTensor(torch.tensor(2)), 0.5 * 2],
            [torch.tensor(0.5), LazyTensor(torch.tensor(2)), 0.5 * 2],
            [LazyTensor(torch.tensor(0.5)), torch.tensor(2), 0.5 * 2],
        ],
    ).check()


def test_scalar_mul_method() -> None:
    Condition(
        lambda a, b, c: scalar_isclose(koila.run(a.mul(b)).item(), c),
        [
            [LazyTensor(torch.tensor(4)), LazyTensor(torch.tensor(3)), 12],
            [torch.tensor(4), LazyTensor(torch.tensor(3)), 12],
            [LazyTensor(torch.tensor(4)), torch.tensor(3), 12],
        ],
    ).check()


def test_scalar_mul_function() -> None:
    Condition(
        lambda a, b, c: scalar_isclose(koila.run(torch.mul(a, b)).item(), c),
        [
            [LazyTensor(torch.tensor(8)), LazyTensor(torch.tensor(4)), 32],
            [torch.tensor(8), LazyTensor(torch.tensor(4)), 32],
            [LazyTensor(torch.tensor(8)), torch.tensor(4), 32],
        ],
    ).check()


def test_scalar_floordiv_op() -> None:
    Condition(
        lambda a, b, c: scalar_isclose(koila.run(a // b).item(), c),
        [
            [LazyTensor(torch.tensor(1)), LazyTensor(torch.tensor(2)), 1 // 2],
            [torch.tensor(1), LazyTensor(torch.tensor(2)), 1 // 2],
            [LazyTensor(torch.tensor(1)), torch.tensor(2), 1 // 2],
        ],
    ).check()


def test_scalar_floordiv_method() -> None:
    Condition(
        lambda a, b, c: scalar_isclose(
            koila.run(a.div(b, rounding_mode="trunc")).item(), c
        ),
        [
            [LazyTensor(torch.tensor(4)), LazyTensor(torch.tensor(3)), 4 // 3],
            [torch.tensor(4), LazyTensor(torch.tensor(3)), 4 // 3],
            [LazyTensor(torch.tensor(4)), torch.tensor(3), 4 // 3],
        ],
    ).check()


def test_scalar_floordiv_function() -> None:
    Condition(
        lambda a, b, c: scalar_isclose(
            koila.run(torch.div(a, b, rounding_mode="trunc")).item(), c
        ),
        [
            [LazyTensor(torch.tensor(9)), LazyTensor(torch.tensor(4)), 9 // 4],
            [torch.tensor(9), LazyTensor(torch.tensor(4)), 9 // 4],
            [LazyTensor(torch.tensor(9)), torch.tensor(4), 9 // 4],
        ],
    ).check()


def test_scalar_truediv_op() -> None:
    Condition(
        lambda a, b, c: scalar_isclose(koila.run(a / b).item(), c),
        [
            [LazyTensor(torch.tensor(1)), LazyTensor(torch.tensor(2)), 1 / 2],
            [torch.tensor(1), LazyTensor(torch.tensor(2)), 1 / 2],
            [LazyTensor(torch.tensor(1)), torch.tensor(2), 1 / 2],
        ],
    ).check()


def test_scalar_truediv_method() -> None:
    Condition(
        lambda a, b, c: scalar_isclose(koila.run(a.div(b)).item(), c),
        [
            [LazyTensor(torch.tensor(4)), LazyTensor(torch.tensor(3)), 4 / 3],
            [torch.tensor(4), LazyTensor(torch.tensor(3)), 4 / 3],
            [LazyTensor(torch.tensor(4)), torch.tensor(3), 4 / 3],
        ],
    ).check()


def test_scalar_truediv_function() -> None:
    Condition(
        lambda a, b, c: scalar_isclose(koila.run(torch.div(a, b)).item(), c),
        [
            [LazyTensor(torch.tensor(9)), LazyTensor(torch.tensor(4)), 9 / 4],
            [torch.tensor(9), LazyTensor(torch.tensor(4)), 9 / 4],
            [LazyTensor(torch.tensor(9)), torch.tensor(4), 9 / 4],
        ],
    ).check()


def test_scalar_pow_op() -> None:
    Condition(
        lambda a, b, c: scalar_isclose(koila.run(a ** b).item(), c),
        [
            [LazyTensor(torch.tensor(1.5)), LazyTensor(torch.tensor(2)), 1.5 ** 2],
            [torch.tensor(1.5), LazyTensor(torch.tensor(2)), 1.5 ** 2],
            [LazyTensor(torch.tensor(1.5)), torch.tensor(2), 1.5 ** 2],
        ],
    ).check()


def test_scalar_pow_method() -> None:
    Condition(
        lambda a, b, c: scalar_isclose(koila.run(a.pow(b)).item(), c),
        [
            [LazyTensor(torch.tensor(4)), LazyTensor(torch.tensor(3)), 4 ** 3],
            [torch.tensor(4), LazyTensor(torch.tensor(3)), 4 ** 3],
            [LazyTensor(torch.tensor(4)), torch.tensor(3), 4 ** 3],
        ],
    ).check()


def test_scalar_pow_function() -> None:
    Condition(
        lambda a, b, c: scalar_isclose(koila.run(torch.pow(a, b)).item(), c),
        [
            [LazyTensor(torch.tensor(9.0)), LazyTensor(torch.tensor(-2)), 9.0 ** -2],
            [torch.tensor(9.0), LazyTensor(torch.tensor(-2)), 9.0 ** -2],
            [LazyTensor(torch.tensor(9.0)), torch.tensor(-2), 9.0 ** -2],
        ],
    ).check()


def test_scalar_remainder_op() -> None:
    Condition(
        lambda a, b, c: scalar_isclose(koila.run(a % b).item(), c),
        [
            [LazyTensor(torch.tensor(3.3)), LazyTensor(torch.tensor(1.9)), 3.3 % 1.9],
            [torch.tensor(3.3), LazyTensor(torch.tensor(1.9)), 3.3 % 1.9],
            [LazyTensor(torch.tensor(3.3)), torch.tensor(1.9), 3.3 % 1.9],
        ],
    ).check()


def test_scalar_remainder_method() -> None:
    Condition(
        lambda a, b, c: scalar_isclose(koila.run(a.remainder(b)).item(), c),
        [
            [LazyTensor(torch.tensor(99)), LazyTensor(torch.tensor(7)), 99 % 7],
            [torch.tensor(99), LazyTensor(torch.tensor(7)), 99 % 7],
            [LazyTensor(torch.tensor(99)), torch.tensor(7), 99 % 7],
        ],
    ).check()


def test_scalar_remainder_function() -> None:
    Condition(
        lambda a, b, c: scalar_isclose(koila.run(torch.remainder(a, b)).item(), c),
        [
            [LazyTensor(torch.tensor(25)), LazyTensor(torch.tensor(7.8)), 25 % 7.8],
            [torch.tensor(25), LazyTensor(torch.tensor(7.8)), 25 % 7.8],
            [LazyTensor(torch.tensor(25)), torch.tensor(7.8), 25 % 7.8],
        ],
    ).check()


def test_matmul_op() -> None:
    arr = torch.randn(2, 10, 11)

    Condition(
        lambda a, b, c: tensor_allclose(koila.run(a @ b), c),
        [
            [LazyTensor(arr[0]), LazyTensor(arr[1].T), arr[0] @ arr[1].T],
            [arr[0], LazyTensor(arr[1].T), arr[0] @ arr[1].T],
            [LazyTensor(arr[0]), arr[1].T, arr[0] @ arr[1].T],
        ],
    ).check()


def test_matmul_method() -> None:
    arr = torch.randn(2, 10, 11)

    Condition(
        lambda a, b, c: tensor_allclose(koila.run(a.matmul(b)), c),
        [
            [LazyTensor(arr[0]), LazyTensor(arr[1].T), arr[0] @ arr[1].T],
            [arr[0], LazyTensor(arr[1].T), arr[0] @ arr[1].T],
            [LazyTensor(arr[0]), arr[1].T, arr[0] @ arr[1].T],
        ],
    ).check()


def test_matmul_function() -> None:
    arr = torch.randn(2, 10, 11)

    Condition(
        lambda a, b, c: tensor_allclose(koila.run(torch.matmul(a, b)), c),
        [
            [LazyTensor(arr[0]), LazyTensor(arr[1].T), arr[0] @ arr[1].T],
            [arr[0], LazyTensor(arr[1].T), arr[0] @ arr[1].T],
            [LazyTensor(arr[0]), arr[1].T, arr[0] @ arr[1].T],
        ],
    ).check()


def test_scalar_identity() -> None:
    tensor = torch.tensor(13.5)

    assert LazyTensor(tensor).run() == 13.5
    assert LazyTensor(tensor).item() == 13.5
    assert int(LazyTensor(tensor)) == 13
    assert float(LazyTensor(tensor)) == 13.5
    assert bool(LazyTensor(tensor))

    tensor = torch.tensor(-17.5)
    assert LazyTensor(tensor).run() == -17.5
    assert LazyTensor(tensor).item() == -17.5
    assert int(LazyTensor(tensor)) == -17
    assert float(LazyTensor(tensor)) == -17.5
    assert bool(LazyTensor(tensor))

    tensor = torch.tensor(0)
    assert not LazyTensor(tensor).run()
    assert not LazyTensor(tensor).item()
    assert not int(LazyTensor(tensor))
    assert not float(LazyTensor(tensor))
    assert not bool(LazyTensor(tensor))


def test_scalar_frac_method() -> None:
    Condition(
        lambda a, c: scalar_isclose(koila.run(a.frac()).item(), c),
        [
            [LazyTensor(torch.tensor(13.22)), 0.22],
            [LazyTensor(torch.tensor(55.0)), 0],
            [LazyTensor(torch.tensor(-55.55)), -0.55],
        ],
    ).check()


def test_scalar_frac_function() -> None:
    Condition(
        lambda a, c: scalar_isclose(koila.run(torch.frac(a)).item(), c),
        [
            [LazyTensor(torch.tensor(25.25)), 0.25],
            [LazyTensor(torch.tensor(11.0)), 0],
            [LazyTensor(torch.tensor(-25.33)), -0.33],
        ],
    ).check()


def test_scalar_exp_method() -> None:
    Condition(
        lambda a, c: scalar_isclose(koila.run(a.exp()).item(), c),
        [
            [LazyTensor(torch.tensor(1.23)), math.e ** 1.23],
            [LazyTensor(torch.tensor(0)), 1],
            [LazyTensor(torch.tensor(1)), math.e],
        ],
    ).check()


def test_scalar_exp_function() -> None:
    Condition(
        lambda a, c: scalar_isclose(koila.run(torch.exp(a)).item(), c),
        [
            [LazyTensor(torch.tensor(0.41)), math.e ** 0.41],
            [LazyTensor(torch.tensor(0)), 1],
            [LazyTensor(torch.tensor(1)), math.e],
        ],
    ).check()


def test_scalar_exp2_method() -> None:
    Condition(
        lambda a, c: scalar_isclose(koila.run(a.exp2()).item(), c),
        [
            [LazyTensor(torch.tensor(10)), 2 ** 10],
            [LazyTensor(torch.tensor(0)), 1],
            [LazyTensor(torch.tensor(1)), 2],
        ],
    ).check()


def test_scalar_exp2_function() -> None:
    Condition(
        lambda a, c: scalar_isclose(koila.run(torch.exp2(a)).item(), c),
        [
            [LazyTensor(torch.tensor(-5)), 2 ** -5],
            [LazyTensor(torch.tensor(0)), 1],
            [LazyTensor(torch.tensor(1)), 2],
        ],
    ).check()


def test_scalar_log_method() -> None:
    Condition(
        lambda a, c: scalar_isclose(koila.run(a.log()).item(), c),
        [
            [LazyTensor(torch.tensor(13)), math.log(13)],
            [LazyTensor(torch.tensor(1)), 0],
            [LazyTensor(torch.tensor(math.e)), 1],
        ],
    ).check()


def test_scalar_log_function() -> None:
    Condition(
        lambda a, c: scalar_isclose(koila.run(torch.log(a)).item(), c),
        [
            [LazyTensor(torch.tensor(5)), math.log(5)],
            [LazyTensor(torch.tensor(1)), 0],
            [LazyTensor(torch.tensor(math.e)), 1],
        ],
    ).check()


def test_scalar_log2_method() -> None:
    Condition(
        lambda a, c: scalar_isclose(koila.run(a.log2()).item(), c),
        [
            [LazyTensor(torch.tensor(442)), math.log2(442)],
            [LazyTensor(torch.tensor(1)), 0],
            [LazyTensor(torch.tensor(2)), 1],
        ],
    ).check()


def test_scalar_log2_function() -> None:
    Condition(
        lambda a, c: scalar_isclose(koila.run(torch.log2(a)).item(), c),
        [
            [LazyTensor(torch.tensor(81)), math.log2(81)],
            [LazyTensor(torch.tensor(1)), 0],
            [LazyTensor(torch.tensor(2)), 1],
        ],
    ).check()


def test_scalar_log10_method() -> None:
    Condition(
        lambda a, c: scalar_isclose(koila.run(a.log10()).item(), c),
        [
            [LazyTensor(torch.tensor(132)), math.log10(132)],
            [LazyTensor(torch.tensor(1)), 0],
            [LazyTensor(torch.tensor(10)), 1],
        ],
    ).check()


def test_scalar_log10_function() -> None:
    Condition(
        lambda a, c: scalar_isclose(koila.run(torch.log10(a)).item(), c),
        [
            [LazyTensor(torch.tensor(979)), math.log10(979)],
            [LazyTensor(torch.tensor(1)), 0],
            [LazyTensor(torch.tensor(10)), 1],
        ],
    ).check()


def test_scalar_log1p_method() -> None:
    Condition(
        lambda a, c: scalar_isclose(koila.run(a.log1p()).item(), c),
        [[LazyTensor(torch.tensor(1.5)), math.log1p(1.5)]],
    ).check()


def test_scalar_log1p_function() -> None:
    Condition(
        lambda a, c: scalar_isclose(koila.run(torch.log1p(a)).item(), c),
        [[LazyTensor(torch.tensor(2.7)), math.log1p(2.7)]],
    ).check()


def test_scalar_abs_method() -> None:
    Condition(
        lambda a, c: scalar_isclose(koila.run(a.abs()).item(), c),
        [
            [LazyTensor(torch.tensor(-1.5)), abs(-1.5)],
            [LazyTensor(torch.tensor(3.7)), abs(3.7)],
        ],
    ).check()


def test_scalar_abs_function() -> None:
    Condition(
        lambda a, c: scalar_isclose(koila.run(torch.abs(a)).item(), c),
        [
            [LazyTensor(torch.tensor(0.001)), abs(0.001)],
            [LazyTensor(torch.tensor(-24)), abs(-24)],
        ],
    ).check()


def test_run_method() -> None:
    random = torch.randn(3, 4, 5, 6)
    Condition(lambda a, b: tensor_allclose(a.run(), b), [[LazyTensor(random), random]])


def test_torch_method() -> None:
    random = torch.randn(3, 4, 5, 6)
    Condition(
        lambda a, b: tensor_allclose(a.torch(), b), [[LazyTensor(random), random]]
    )


def test_numpy_method() -> None:
    random = torch.randn(3, 4, 5, 6)
    Condition(
        lambda a, b: array_allclose(a.numpy(), b.numpy()),
        [[LazyTensor(random), random]],
    )
