# Copyright (c) RenChu Wang - All Rights Reserved

import math
import typing

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

import koila
from koila import Evaluation, LazyTensor, Runnable, RunnableTensor

from . import common


def test_lazytensor_is_runnable() -> None:
    assert issubclass(Evaluation, Runnable)
    assert issubclass(Evaluation, RunnableTensor)
    assert issubclass(LazyTensor, Runnable)
    assert issubclass(LazyTensor, RunnableTensor)


def test_positive_op() -> None:
    common.call(
        lambda a, c: common.assert_isclose((+a).item(), c),
        [[LazyTensor(torch.tensor(-11)), -11]],
    )


def test_positive_method() -> None:
    common.call(
        lambda a, c: common.assert_isclose(a.positive().item(), c),
        [[LazyTensor(torch.tensor(4)), 4]],
    )


def test_positive_function() -> None:
    common.call(
        lambda a, c: common.assert_isclose(torch.positive(a).item(), c),
        [[LazyTensor(torch.tensor(-8)), -8]],
    )


def test_negative_op() -> None:
    common.call(
        lambda a, c: common.assert_isclose((-a).item(), c),
        [[LazyTensor(torch.tensor(-13)), 13]],
    )


def test_negative_method() -> None:
    common.call(
        lambda a, c: common.assert_isclose(a.neg().item(), c),
        [[LazyTensor(torch.tensor(2)), -2]],
    )


def test_negative_function() -> None:
    common.call(
        lambda a, c: common.assert_equal(torch.neg(a).item(), c),
        [[LazyTensor(torch.tensor(-5)), 5]],
    )


def test_eq_ne_op() -> None:
    arr = torch.randint(0, 2, [2, 3, 4])
    brr = torch.randint(0, 2, [2, 3, 4])
    la = typing.cast(Tensor, LazyTensor(arr))
    lb = typing.cast(Tensor, LazyTensor(brr))
    common.call(
        lambda a, c: common.assert_equal(koila.run(a), c),
        [[la == lb, arr == brr], [la != lb, arr != brr]],
    )


def test_cmp_op() -> None:
    arr = torch.randint(0, 5, [2, 3, 4])
    brr = torch.randint(0, 5, [2, 3, 4])
    la = typing.cast(Tensor, LazyTensor(arr))
    lb = typing.cast(Tensor, LazyTensor(brr))
    common.call(
        lambda a, c: common.assert_equal(koila.run(a), c),
        [
            [la < lb, arr < brr],
            [la <= lb, arr <= brr],
            [la > lb, arr > brr],
            [la >= lb, arr >= brr],
        ],
    )


def test_add_op() -> None:
    common.call(
        lambda a, b, c: common.assert_isclose((a + b).item(), c),
        [
            [LazyTensor(torch.tensor(1)), LazyTensor(torch.tensor(2)), 1 + 2],
            [torch.tensor(1), LazyTensor(torch.tensor(2)), 1 + 2],
            [LazyTensor(torch.tensor(1)), torch.tensor(2), 1 + 2],
        ],
    )


def test_add_method() -> None:
    common.call(
        lambda a, b, c: common.assert_isclose(a.add(b).item(), c),
        [
            [LazyTensor(torch.tensor(4)), LazyTensor(torch.tensor(3)), 4 + 3],
            [torch.tensor(4), LazyTensor(torch.tensor(3)), 4 + 3],
            [LazyTensor(torch.tensor(4)), torch.tensor(3), 4 + 3],
        ],
    )


def test_add_function() -> None:
    common.call(
        lambda a, b, c: common.assert_isclose(torch.add(a, b).item(), c),
        [
            [LazyTensor(torch.tensor(8)), LazyTensor(torch.tensor(4)), 8 + 4],
            [torch.tensor(8), LazyTensor(torch.tensor(4)), 8 + 4],
            [LazyTensor(torch.tensor(8)), torch.tensor(4), 8 + 4],
        ],
    )


def test_sub_op() -> None:
    common.call(
        lambda a, b, c: common.assert_isclose((a - b).item(), c),
        [
            [LazyTensor(torch.tensor(1)), LazyTensor(torch.tensor(2)), 1 - 2],
            [torch.tensor(1), LazyTensor(torch.tensor(2)), 1 - 2],
            [LazyTensor(torch.tensor(1)), torch.tensor(2), 1 - 2],
        ],
    )


def test_sub_method() -> None:
    common.call(
        lambda a, b, c: common.assert_isclose(a.sub(b).item(), c),
        [
            [LazyTensor(torch.tensor(4)), LazyTensor(torch.tensor(3)), 4 - 3],
            [torch.tensor(4), LazyTensor(torch.tensor(3)), 4 - 3],
            [LazyTensor(torch.tensor(4)), torch.tensor(3), 4 - 3],
        ],
    )


def test_sub_function() -> None:
    common.call(
        lambda a, b, c: common.assert_isclose(torch.sub(a, b).item(), c),
        [
            [LazyTensor(torch.tensor(8)), LazyTensor(torch.tensor(4)), 8 - 4],
            [torch.tensor(8), LazyTensor(torch.tensor(4)), 8 - 4],
            [LazyTensor(torch.tensor(8)), torch.tensor(4), 8 - 4],
        ],
    )


def test_mul_op() -> None:
    common.call(
        lambda a, b, c: common.assert_isclose((a * b).item(), c),
        [
            [LazyTensor(torch.tensor(0.5)), LazyTensor(torch.tensor(2)), 0.5 * 2],
            [torch.tensor(0.5), LazyTensor(torch.tensor(2)), 0.5 * 2],
            [LazyTensor(torch.tensor(0.5)), torch.tensor(2), 0.5 * 2],
        ],
    )


def test_mul_method() -> None:
    common.call(
        lambda a, b, c: common.assert_isclose(a.mul(b).item(), c),
        [
            [LazyTensor(torch.tensor(4)), LazyTensor(torch.tensor(3)), 12],
            [torch.tensor(4), LazyTensor(torch.tensor(3)), 12],
            [LazyTensor(torch.tensor(4)), torch.tensor(3), 12],
        ],
    )


def test_mul_function() -> None:
    common.call(
        lambda a, b, c: common.assert_isclose(torch.mul(a, b).item(), c),
        [
            [LazyTensor(torch.tensor(8)), LazyTensor(torch.tensor(4)), 32],
            [torch.tensor(8), LazyTensor(torch.tensor(4)), 32],
            [LazyTensor(torch.tensor(8)), torch.tensor(4), 32],
        ],
    )


def test_floordiv_op() -> None:
    common.call(
        common.is_notimplemented,
        [
            [lambda: LazyTensor(torch.tensor(1)) // LazyTensor(torch.tensor(2))],
            [lambda: torch.tensor(1) // LazyTensor(torch.tensor(2))],
            [lambda: LazyTensor(torch.tensor(1)) // torch.tensor(2)],
        ],
    )


def test_floordiv_method() -> None:
    common.call(
        lambda a, b, c: common.assert_isclose(
            a.div(b, rounding_mode="trunc").item(), c
        ),
        [
            [LazyTensor(torch.tensor(4)), LazyTensor(torch.tensor(3)), 4 // 3],
            [torch.tensor(4), LazyTensor(torch.tensor(3)), 4 // 3],
            [LazyTensor(torch.tensor(4)), torch.tensor(3), 4 // 3],
        ],
    )


def test_floordiv_function() -> None:
    common.call(
        lambda a, b, c: common.assert_isclose(
            torch.div(a, b, rounding_mode="trunc").item(), c
        ),
        [
            [LazyTensor(torch.tensor(9)), LazyTensor(torch.tensor(4)), 9 // 4],
            [torch.tensor(9), LazyTensor(torch.tensor(4)), 9 // 4],
            [LazyTensor(torch.tensor(9)), torch.tensor(4), 9 // 4],
        ],
    )


def test_truediv_op() -> None:
    common.call(
        lambda a, b, c: common.assert_isclose((a / b).item(), c),
        [
            [LazyTensor(torch.tensor(1)), LazyTensor(torch.tensor(2)), 1 / 2],
            [torch.tensor(1), LazyTensor(torch.tensor(2)), 1 / 2],
            [LazyTensor(torch.tensor(1)), torch.tensor(2), 1 / 2],
        ],
    )


def test_truediv_method() -> None:
    common.call(
        lambda a, b, c: common.assert_isclose(a.div(b).item(), c),
        [
            [LazyTensor(torch.tensor(4)), LazyTensor(torch.tensor(3)), 4 / 3],
            [torch.tensor(4), LazyTensor(torch.tensor(3)), 4 / 3],
            [LazyTensor(torch.tensor(4)), torch.tensor(3), 4 / 3],
        ],
    )


def test_truediv_function() -> None:
    common.call(
        lambda a, b, c: common.assert_isclose(torch.div(a, b).item(), c),
        [
            [LazyTensor(torch.tensor(9)), LazyTensor(torch.tensor(4)), 9 / 4],
            [torch.tensor(9), LazyTensor(torch.tensor(4)), 9 / 4],
            [LazyTensor(torch.tensor(9)), torch.tensor(4), 9 / 4],
        ],
    )


def test_pow_op() -> None:
    common.call(
        lambda a, b, c: common.assert_isclose((a**b).item(), c),
        [
            [LazyTensor(torch.tensor(1.5)), LazyTensor(torch.tensor(2)), 1.5**2],
            [torch.tensor(1.5), LazyTensor(torch.tensor(2)), 1.5**2],
            [LazyTensor(torch.tensor(1.5)), torch.tensor(2), 1.5**2],
        ],
    )


def test_pow_method() -> None:
    common.call(
        lambda a, b, c: common.assert_isclose(a.pow(b).item(), c),
        [
            [LazyTensor(torch.tensor(4)), LazyTensor(torch.tensor(3)), 4**3],
            [torch.tensor(4), LazyTensor(torch.tensor(3)), 4**3],
            [LazyTensor(torch.tensor(4)), torch.tensor(3), 4**3],
        ],
    )


def test_pow_function() -> None:
    common.call(
        lambda a, b, c: common.assert_isclose(torch.pow(a, b).item(), c),
        [
            [LazyTensor(torch.tensor(9.0)), LazyTensor(torch.tensor(-2)), 9.0**-2],
            [torch.tensor(9.0), LazyTensor(torch.tensor(-2)), 9.0**-2],
            [LazyTensor(torch.tensor(9.0)), torch.tensor(-2), 9.0**-2],
        ],
    )


def test_remainder_op() -> None:
    common.call(
        lambda a, b, c: common.assert_isclose((a % b).item(), c),
        [
            [LazyTensor(torch.tensor(3.3)), LazyTensor(torch.tensor(1.9)), 3.3 % 1.9],
            [torch.tensor(3.3), LazyTensor(torch.tensor(1.9)), 3.3 % 1.9],
            [LazyTensor(torch.tensor(3.3)), torch.tensor(1.9), 3.3 % 1.9],
        ],
    )


def test_remainder_method() -> None:
    common.call(
        lambda a, b, c: common.assert_isclose(a.remainder(b).item(), c),
        [
            [LazyTensor(torch.tensor(99)), LazyTensor(torch.tensor(7)), 99 % 7],
            [torch.tensor(99), LazyTensor(torch.tensor(7)), 99 % 7],
            [LazyTensor(torch.tensor(99)), torch.tensor(7), 99 % 7],
        ],
    )


def test_remainder_function() -> None:
    common.call(
        lambda a, b, c: common.assert_isclose(torch.remainder(a, b).item(), c),
        [
            [LazyTensor(torch.tensor(25)), LazyTensor(torch.tensor(7.8)), 25 % 7.8],
            [torch.tensor(25), LazyTensor(torch.tensor(7.8)), 25 % 7.8],
            [LazyTensor(torch.tensor(25)), torch.tensor(7.8), 25 % 7.8],
        ],
    )


def test_matmul_op() -> None:
    arr = torch.randn(2, 10, 11)

    common.call(
        lambda a, b, c: common.assert_isclose(koila.run(a @ b), c),
        [
            [LazyTensor(arr[0]), LazyTensor(arr[1].T), arr[0] @ arr[1].T],
            [arr[0], LazyTensor(arr[1].T), arr[0] @ arr[1].T],
            [LazyTensor(arr[0]), arr[1].T, arr[0] @ arr[1].T],
        ],
    )


def test_matmul_method() -> None:
    arr = torch.randn(2, 10, 11)

    common.call(
        lambda a, b, c: common.assert_isclose(koila.run(a.matmul(b)), c),
        [
            [LazyTensor(arr[0]), LazyTensor(arr[1].T), arr[0] @ arr[1].T],
            [arr[0], LazyTensor(arr[1].T), arr[0] @ arr[1].T],
            [LazyTensor(arr[0]), arr[1].T, arr[0] @ arr[1].T],
        ],
    )


def test_matmul_function() -> None:
    arr = torch.randn(2, 10, 11)

    common.call(
        lambda a, b, c: common.assert_isclose(koila.run(torch.matmul(a, b)), c),
        [
            [LazyTensor(arr[0]), LazyTensor(arr[1].T), arr[0] @ arr[1].T],
            [arr[0], LazyTensor(arr[1].T), arr[0] @ arr[1].T],
            [LazyTensor(arr[0]), arr[1].T, arr[0] @ arr[1].T],
        ],
    )


def test_identity() -> None:
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


def test_frac_method() -> None:
    common.call(
        lambda a, c: common.assert_isclose(a.frac().item(), c),
        [
            [LazyTensor(torch.tensor(13.22)), 0.22],
            [LazyTensor(torch.tensor(55.0)), 0],
            [LazyTensor(torch.tensor(-55.55)), -0.55],
        ],
    )


def test_frac_function() -> None:
    common.call(
        lambda a, c: common.assert_isclose(torch.frac(a).item(), c),
        [
            [LazyTensor(torch.tensor(25.25)), 0.25],
            [LazyTensor(torch.tensor(11.0)), 0],
            [LazyTensor(torch.tensor(-25.33)), -0.33],
        ],
    )


def test_exp_method() -> None:
    common.call(
        lambda a, c: common.assert_isclose(a.exp().item(), c),
        [
            [LazyTensor(torch.tensor(1.23)), math.e**1.23],
            [LazyTensor(torch.tensor(0)), 1],
            [LazyTensor(torch.tensor(1)), math.e],
        ],
    )


def test_exp_function() -> None:
    common.call(
        lambda a, c: common.assert_isclose(torch.exp(a).item(), c),
        [
            [LazyTensor(torch.tensor(0.41)), math.e**0.41],
            [LazyTensor(torch.tensor(0)), 1],
            [LazyTensor(torch.tensor(1)), math.e],
        ],
    )


def test_exp2_method() -> None:
    common.call(
        lambda a, c: common.assert_isclose(a.exp2().item(), c),
        [
            [LazyTensor(torch.tensor(10)), 2**10],
            [LazyTensor(torch.tensor(0)), 1],
            [LazyTensor(torch.tensor(1)), 2],
        ],
    )


def test_exp2_function() -> None:
    common.call(
        lambda a, c: common.assert_isclose(torch.exp2(a).item(), c),
        [
            [LazyTensor(torch.tensor(-5)), 2**-5],
            [LazyTensor(torch.tensor(0)), 1],
            [LazyTensor(torch.tensor(1)), 2],
        ],
    )


def test_log_method() -> None:
    common.call(
        lambda a, c: common.assert_isclose(a.log().item(), c),
        [
            [LazyTensor(torch.tensor(13)), math.log(13)],
            [LazyTensor(torch.tensor(1)), 0],
            [LazyTensor(torch.tensor(math.e)), 1],
        ],
    )


def test_log_function() -> None:
    common.call(
        lambda a, c: common.assert_isclose(torch.log(a).item(), c),
        [
            [LazyTensor(torch.tensor(5)), math.log(5)],
            [LazyTensor(torch.tensor(1)), 0],
            [LazyTensor(torch.tensor(math.e)), 1],
        ],
    )


def test_log2_method() -> None:
    common.call(
        lambda a, c: common.assert_isclose(a.log2().item(), c),
        [
            [LazyTensor(torch.tensor(442)), math.log2(442)],
            [LazyTensor(torch.tensor(1)), 0],
            [LazyTensor(torch.tensor(2)), 1],
        ],
    )


def test_log2_function() -> None:
    common.call(
        lambda a, c: common.assert_isclose(torch.log2(a).item(), c),
        [
            [LazyTensor(torch.tensor(81)), math.log2(81)],
            [LazyTensor(torch.tensor(1)), 0],
            [LazyTensor(torch.tensor(2)), 1],
        ],
    )


def test_log10_method() -> None:
    common.call(
        lambda a, c: common.assert_isclose(a.log10().item(), c),
        [
            [LazyTensor(torch.tensor(132)), math.log10(132)],
            [LazyTensor(torch.tensor(1)), 0],
            [LazyTensor(torch.tensor(10)), 1],
        ],
    )


def test_log10_function() -> None:
    common.call(
        lambda a, c: common.assert_isclose(torch.log10(a).item(), c),
        [
            [LazyTensor(torch.tensor(979)), math.log10(979)],
            [LazyTensor(torch.tensor(1)), 0],
            [LazyTensor(torch.tensor(10)), 1],
        ],
    )


def test_log1p_method() -> None:
    common.call(
        lambda a, c: common.assert_isclose(a.log1p().item(), c),
        [[LazyTensor(torch.tensor(1.5)), math.log1p(1.5)]],
    )


def test_log1p_function() -> None:
    common.call(
        lambda a, c: common.assert_isclose(torch.log1p(a).item(), c),
        [[LazyTensor(torch.tensor(2.7)), math.log1p(2.7)]],
    )


def test_abs_op() -> None:
    common.call(
        lambda a, c: common.assert_isclose(abs(a).item(), c),
        [
            [LazyTensor(torch.tensor(-7.122)), abs(-7.122)],
            [LazyTensor(torch.tensor(4.002)), abs(4.002)],
        ],
    )


def test_abs_method() -> None:
    common.call(
        lambda a, c: common.assert_isclose(a.abs().item(), c),
        [
            [LazyTensor(torch.tensor(-1.5)), abs(-1.5)],
            [LazyTensor(torch.tensor(3.7)), abs(3.7)],
        ],
    )


def test_abs_function() -> None:
    common.call(
        lambda a, c: common.assert_isclose(torch.abs(a).item(), c),
        [
            [LazyTensor(torch.tensor(0.001)), abs(0.001)],
            [LazyTensor(torch.tensor(-24)), abs(-24)],
        ],
    )


def test_min_method() -> None:
    arr = torch.randn(6, 7, 8)

    common.call(
        lambda a, c: common.assert_isclose(koila.run(a), c),
        [
            [LazyTensor(arr).min(), arr.min()],
            [LazyTensor(arr).min(1)[0], arr.min(1)[0]],
            [LazyTensor(arr).min(1)[1], arr.min(1)[1]],
        ],
    )


def test_min_function() -> None:
    arr = torch.randn(6, 7, 8)
    brr = torch.randn(1, 7, 8)
    la = typing.cast(Tensor, LazyTensor(arr))
    lb = typing.cast(Tensor, LazyTensor(brr))

    common.call(
        lambda a, c: common.assert_isclose(koila.run(a), c),
        [
            [torch.min(la), torch.min(arr)],
            [torch.min(la, 2)[0], torch.min(arr, 2)[0]],
            [
                torch.min(la, 1, keepdim=True).indices,
                torch.min(arr, 1, keepdim=True).indices,
            ],
            [torch.min(la, lb), torch.min(arr, brr)],
        ],
    )


def test_max_method() -> None:
    arr = torch.randn(6, 7, 8)

    common.call(
        lambda a, c: common.assert_isclose(koila.run(a), c),
        [
            [LazyTensor(arr).max(), arr.max()],
            [LazyTensor(arr).max(1)[0], arr.max(1)[0]],
            [LazyTensor(arr).max(1)[1], arr.max(1)[1]],
        ],
    )


def test_max_function() -> None:
    arr = torch.randn(6, 7, 8)
    brr = torch.randn(1, 7, 8)
    la = typing.cast(Tensor, LazyTensor(arr))
    lb = typing.cast(Tensor, LazyTensor(brr))

    common.call(
        lambda a, c: common.assert_isclose(koila.run(a), c),
        [
            [torch.max(la), torch.max(arr)],
            [torch.max(la, 2)[0], torch.max(arr, 2)[0]],
            [
                torch.max(la, 1, keepdim=True).indices,
                torch.max(arr, 1, keepdim=True).indices,
            ],
            [torch.max(la, lb), torch.max(arr, brr)],
        ],
    )


def test_size_shape_method() -> None:
    arr = torch.randn(11, 13)
    la = LazyTensor(arr)
    assert la.size() == la.shape == (11, 13)
    assert la.size(0) == 11
    assert la.size(1) == 13


def test_t_method() -> None:
    arr = torch.randn(11, 13)
    la = LazyTensor(arr)
    assert la.T.size() == la.t().size() == (13, 11)


def test_t_function() -> None:
    arr = torch.randn(11, 13)
    la = typing.cast(Tensor, LazyTensor(arr))
    assert torch.t(la).shape == (13, 11)


def test_dim_method() -> None:
    arr = torch.randn(11, 13)
    assert arr.ndim == arr.dim() == 2
    arr = torch.randn(1, 2, 3, 4, 5)
    assert arr.dim() == 5


def test_permute_method() -> None:
    arr = torch.randn(2, 3, 4, 5, 6)
    la = LazyTensor(arr)
    assert la.permute(3, 4, 1, 2, 0).shape == (5, 6, 3, 4, 2)
    assert la.permute(0, 1, 4, 3, 2).shape == (2, 3, 6, 5, 4)


def test_permute_function() -> None:
    arr = torch.randn(2, 3, 4, 5, 6)
    la = typing.cast(Tensor, LazyTensor(arr))
    assert torch.permute(la, (3, 4, 1, 2, 0)).shape == (5, 6, 3, 4, 2)
    assert torch.permute(la, (0, 1, 4, 3, 2)).shape == (2, 3, 6, 5, 4)


def test_transpose_method() -> None:
    arr = torch.randn(2, 3, 4, 5, 6)
    la = LazyTensor(arr)
    assert la.transpose(3, 4).shape == (2, 3, 4, 6, 5)
    assert la.transpose(0, 1).shape == (3, 2, 4, 5, 6)
    assert la.transpose(0, 3).shape == (5, 3, 4, 2, 6)


def test_select_method() -> None:
    arr = torch.randn(3, 4, 5)
    sel = arr.select(1, 2)
    assert isinstance(sel, Tensor)
    assert not isinstance(sel, LazyTensor)

    la = LazyTensor(arr)
    lsel = la.select(1, 2)

    assert not isinstance(lsel, Tensor)
    assert isinstance(lsel, LazyTensor)
    assert sel.size() == lsel.size() == (3, 5)
    common.assert_isclose(lsel.run(), sel)


def test_select_function() -> None:
    arr = torch.randn(3, 4, 5)
    sel = torch.select(arr, 1, 2)
    assert isinstance(sel, Tensor)
    assert not isinstance(sel, LazyTensor)

    la = typing.cast(Tensor, LazyTensor(arr))
    lsel = torch.select(la, 1, 2)

    assert not isinstance(lsel, Tensor)
    assert isinstance(lsel, LazyTensor)
    assert sel.size() == lsel.size() == (3, 5)
    common.assert_isclose(lsel.run(), sel)


def test_index_select_method() -> None:
    arr = torch.randn(3, 4, 5)
    idx = torch.tensor([1, 2, 3])
    sel = arr.index_select(1, idx)
    assert isinstance(sel, Tensor)
    assert not isinstance(sel, LazyTensor)

    la = LazyTensor(arr)
    lsel = la.index_select(1, idx)

    assert not isinstance(lsel, Tensor)
    assert isinstance(lsel, LazyTensor)
    assert sel.size() == lsel.size() == (3, 3, 5)
    common.assert_isclose(lsel.run(), sel)


def test_index_select_function() -> None:
    arr = torch.randn(3, 4, 5)
    idx = torch.tensor([1, 2, 3])
    sel = torch.index_select(arr, 1, idx)
    assert isinstance(sel, Tensor)
    assert not isinstance(sel, LazyTensor)

    la = typing.cast(Tensor, LazyTensor(arr))
    lsel = torch.index_select(la, 1, idx)

    assert not isinstance(lsel, Tensor)
    assert isinstance(lsel, LazyTensor)
    assert sel.size() == lsel.size() == (3, 3, 5)
    common.assert_isclose(lsel.run(), sel)


def test_numel_method() -> None:
    arr = torch.randn(2, 3, 4, 5, 6)
    la = typing.cast(Tensor, LazyTensor(arr))
    assert la.numel() == 2 * 3 * 4 * 5 * 6

    arr = torch.randn(15, 19)
    la = typing.cast(Tensor, LazyTensor(arr))
    assert la.numel() == 15 * 19


def test_numel_function() -> None:
    arr = torch.randn(2, 3, 4, 5, 6)
    la = typing.cast(Tensor, LazyTensor(arr))
    assert torch.numel(la) == 2 * 3 * 4 * 5 * 6

    arr = torch.randn(15, 19)
    la = typing.cast(Tensor, LazyTensor(arr))
    assert torch.numel(la) == 15 * 19


def test_sigmoid_method() -> None:
    arr = torch.randn(4, 5, 6)
    common.call(
        lambda a, c: common.assert_isclose(koila.run(a), c),
        [[LazyTensor(arr).sigmoid(), torch.sigmoid(arr)]],
    )


def test_sigmoid_function() -> None:
    arr = torch.randn(4, 5, 6)
    la = typing.cast(Tensor, arr)
    common.call(
        lambda a, c: common.assert_isclose(koila.run(a), c),
        [[torch.sigmoid(la), torch.sigmoid(arr)]],
    )


def test_sin_method() -> None:
    common.call(
        lambda a, c: common.assert_isclose(a.sin().item(), c),
        [
            [LazyTensor(torch.tensor(0)), 0],
            [LazyTensor(torch.tensor(math.pi)), 0],
            [LazyTensor(torch.tensor(math.pi / 2)), 1],
            [LazyTensor(torch.tensor(3 * math.pi / 2)), -1],
            [LazyTensor(torch.tensor(42.0)), math.sin(42)],
            [LazyTensor(torch.tensor(-75.0)), math.sin(-75)],
        ],
    )


def test_sin_function() -> None:
    common.call(
        lambda a, c: common.assert_isclose(torch.sin(a).item(), c),
        [
            [LazyTensor(torch.tensor(0)), 0],
            [LazyTensor(torch.tensor(math.pi)), 0],
            [LazyTensor(torch.tensor(math.pi / 2)), 1],
            [LazyTensor(torch.tensor(3 * math.pi / 2)), -1],
            [LazyTensor(torch.tensor(42.0)), math.sin(42)],
            [LazyTensor(torch.tensor(-75.0)), math.sin(-75)],
        ],
    )


def test_cos_method() -> None:
    common.call(
        lambda a, c: common.assert_isclose(a.cos().item(), c),
        [
            [LazyTensor(torch.tensor(0)), 1],
            [LazyTensor(torch.tensor(math.pi)), -1],
            [LazyTensor(torch.tensor(math.pi / 2)), 0],
            [LazyTensor(torch.tensor(3 * math.pi / 2)), 0],
            [LazyTensor(torch.tensor(27.0)), math.cos(27)],
            [LazyTensor(torch.tensor(-14.0)), math.cos(-14)],
        ],
    )


def test_cos_function() -> None:
    common.call(
        lambda a, c: common.assert_isclose(torch.cos(a).item(), c),
        [
            [LazyTensor(torch.tensor(0)), 1],
            [LazyTensor(torch.tensor(math.pi)), -1],
            [LazyTensor(torch.tensor(math.pi / 2)), 0],
            [LazyTensor(torch.tensor(3 * math.pi / 2)), 0],
            [LazyTensor(torch.tensor(27.0)), math.cos(27)],
            [LazyTensor(torch.tensor(-14.0)), math.cos(-14)],
        ],
    )


def test_tan_method() -> None:
    common.call(
        lambda a, c: common.assert_isclose(a.tan().item(), c),
        [
            [LazyTensor(torch.tensor(0)), 0],
            [LazyTensor(torch.tensor(math.pi)), 0],
            [LazyTensor(torch.tensor(99.0)), math.tan(99)],
            [LazyTensor(torch.tensor(-4.0)), math.tan(-4)],
        ],
    )


def test_tan_function() -> None:
    common.call(
        lambda a, c: common.assert_isclose(torch.tan(a).item(), c),
        [
            [LazyTensor(torch.tensor(0)), 0],
            [LazyTensor(torch.tensor(math.pi)), 0],
            [LazyTensor(torch.tensor(99.0)), math.tan(99)],
            [LazyTensor(torch.tensor(-4.0)), math.tan(-4)],
        ],
    )


def test_asin_method() -> None:
    common.call(
        lambda a, c: common.assert_isclose(a.asin().item(), c),
        [
            [LazyTensor(torch.tensor(n)), math.asin(n)]
            for n in np.linspace(-1, 1).tolist()
        ],
    )


def test_asin_function() -> None:
    common.call(
        lambda a, c: common.assert_isclose(torch.asin(a).item(), c),
        [
            [LazyTensor(torch.tensor(n)), math.asin(n)]
            for n in np.linspace(-1, 1).tolist()
        ],
    )


def test_acos_method() -> None:
    common.call(
        lambda a, c: common.assert_isclose(a.acos().item(), c),
        [
            [LazyTensor(torch.tensor(n)), math.acos(n)]
            for n in np.linspace(-1, 1).tolist()
        ],
    )


def test_acos_function() -> None:
    common.call(
        lambda a, c: common.assert_isclose(torch.acos(a).item(), c),
        [
            [LazyTensor(torch.tensor(n)), math.acos(n)]
            for n in np.linspace(-1, 1).tolist()
        ],
    )


def test_atan_method() -> None:
    common.call(
        lambda a, c: common.assert_isclose(a.atan().item(), c),
        [
            [LazyTensor(torch.tensor(99.0)), math.atan(99)],
            [LazyTensor(torch.tensor(-4.0)), math.atan(-4)],
            [LazyTensor(torch.tensor(-6.0)), math.atan(-6)],
            [LazyTensor(torch.tensor(242.0)), math.atan(242)],
        ],
    )


def test_atan_function() -> None:
    common.call(
        lambda a, c: common.assert_isclose(torch.atan(a).item(), c),
        [
            [LazyTensor(torch.tensor(99.0)), math.atan(99)],
            [LazyTensor(torch.tensor(-4.0)), math.atan(-4)],
            [LazyTensor(torch.tensor(-6.0)), math.atan(-6)],
            [LazyTensor(torch.tensor(242.0)), math.atan(242)],
        ],
    )


def test_sinh_method() -> None:
    common.call(
        lambda a, c: common.assert_isclose(a.sinh().item(), c),
        [
            [LazyTensor(torch.tensor(n)), math.sinh(n)]
            for n in np.linspace(-1, 1).tolist()
        ],
    )


def test_sinh_function() -> None:
    common.call(
        lambda a, c: common.assert_isclose(torch.sinh(a).item(), c),
        [
            [LazyTensor(torch.tensor(n)), math.sinh(n)]
            for n in np.linspace(-1, 1).tolist()
        ],
    )


def test_cosh_method() -> None:
    common.call(
        lambda a, c: common.assert_isclose(a.cosh().item(), c),
        [
            [LazyTensor(torch.tensor(n)), math.cosh(n)]
            for n in np.linspace(-1, 1).tolist()
        ],
    )


def test_cosh_function() -> None:
    common.call(
        lambda a, c: common.assert_isclose(torch.cosh(a).item(), c),
        [
            [LazyTensor(torch.tensor(n)), math.cosh(n)]
            for n in np.linspace(-1, 1).tolist()
        ],
    )


def test_tanh_method() -> None:
    common.call(
        lambda a, c: common.assert_isclose(a.tanh().item(), c),
        [[LazyTensor(torch.tensor(n)), math.tanh(n)] for n in np.linspace(-10, 10)],
    )


def test_tanh_function() -> None:
    common.call(
        lambda a, c: common.assert_isclose(torch.tanh(a).item(), c),
        [[LazyTensor(torch.tensor(n)), math.tanh(n)] for n in np.linspace(-10, 10)],
    )


def test_asinh_method() -> None:
    common.call(
        lambda a, c: common.assert_isclose(a.asinh().item(), c),
        [
            [LazyTensor(torch.tensor(199.0)), math.asinh(199)],
            [LazyTensor(torch.tensor(-241.0)), math.asinh(-241)],
            [LazyTensor(torch.tensor(-9.0)), math.asinh(-9)],
            [LazyTensor(torch.tensor(0.0)), math.asinh(0)],
        ],
    )


def test_asinh_function() -> None:
    common.call(
        lambda a, c: common.assert_isclose(torch.asinh(a).item(), c),
        [
            [LazyTensor(torch.tensor(199.0)), math.asinh(199)],
            [LazyTensor(torch.tensor(-241.0)), math.asinh(-241)],
            [LazyTensor(torch.tensor(-9.0)), math.asinh(-9)],
            [LazyTensor(torch.tensor(0.0)), math.asinh(0)],
        ],
    )


def test_acosh_method() -> None:
    common.call(
        lambda a, c: common.assert_isclose(a.acosh().item(), c),
        [
            [LazyTensor(torch.tensor(14.0)), math.acosh(14)],
            [LazyTensor(torch.tensor(2.0)), math.acosh(2)],
            [LazyTensor(torch.tensor(1.0)), math.acosh(1)],
            [LazyTensor(torch.tensor(65.0)), math.acosh(65)],
        ],
    )


def test_acosh_function() -> None:
    common.call(
        lambda a, c: common.assert_isclose(torch.acosh(a).item(), c),
        [
            [LazyTensor(torch.tensor(14.0)), math.acosh(14)],
            [LazyTensor(torch.tensor(2.0)), math.acosh(2)],
            [LazyTensor(torch.tensor(1.0)), math.acosh(1)],
            [LazyTensor(torch.tensor(65.0)), math.acosh(65)],
        ],
    )


def test_atanh_method() -> None:
    common.call(
        lambda a, c: common.assert_isclose(a.atanh().item(), c),
        [
            [LazyTensor(torch.tensor(n)), math.atanh(n)]
            for n in np.linspace(-0.99, 0.99, endpoint=False).tolist()
        ],
    )


def test_atanh_function() -> None:
    common.call(
        lambda a, c: common.assert_isclose(torch.atanh(a).item(), c),
        [
            [LazyTensor(torch.tensor(n)), math.atanh(n)]
            for n in np.linspace(-0.99, 0.99, endpoint=False).tolist()
        ],
    )


def test_run_method() -> None:
    random = torch.randn(3, 4, 5, 6)
    common.call(
        lambda a, b: common.assert_isclose(a.run(), b), [[LazyTensor(random), random]]
    )


def test_torch_method() -> None:
    random = torch.randn(3, 4, 5, 6)
    common.call(
        lambda a, b: common.assert_isclose(a.torch(), b), [[LazyTensor(random), random]]
    )


def test_numpy_method() -> None:
    random = torch.randn(3, 4, 5, 6)
    common.call(
        lambda a, b: common.assert_isclose(a.numpy(), b.numpy()),
        [[LazyTensor(random), random]],
    )


def test_pad_function() -> None:
    tensor = torch.randn(3, 4, 5, 6)
    padded = F.pad(tensor, (2, 3, 0, 1), mode="reflect")
    assert isinstance(padded, Tensor)
    assert not isinstance(padded, LazyTensor)

    la = typing.cast(Tensor, LazyTensor(tensor))
    lazy_padded = F.pad(la, (2, 3, 0, 1), mode="reflect")
    assert not isinstance(lazy_padded, Tensor)
    assert isinstance(lazy_padded, LazyTensor)
    assert padded.shape == lazy_padded.shape

    common.assert_isclose(lazy_padded.run(), padded)


def test_buffer_sizes() -> None:
    a = torch.randn(4, 5, 6)

    la = LazyTensor(a)
    assert a.numel() == la.numel() == la.buffer_numel()[1]

    b = torch.randn(4, 5, 1)
    lb = LazyTensor(b)
    assert b.numel() == lb.numel() == lb.buffer_numel()[1]

    lc = typing.cast(LazyTensor, la + lb)
    assert lc.numel() == la.numel() == 6 * lb.numel()
    assert lc.buffer_numel()[1] == la.numel() + lb.numel() + lc.numel()

    d = torch.randn(4, 5, 6)
    ld = typing.cast(LazyTensor, d)

    le = typing.cast(LazyTensor, lc * ld)
    assert d.numel() == ld.numel() == le.numel()
    assert le.buffer_numel()[1] == sum(map(LazyTensor.numel, {la, lb, lc, ld, le}))

    lf = le.sum()
    assert lf.buffer_numel()[1] == sum(map(LazyTensor.numel, {la, lb, lc, ld, le, lf}))

    lg = typing.cast(LazyTensor, lc + le)
    assert lg.buffer_numel()[1] == sum(map(LazyTensor.numel, {la, lb, lc, ld, le, lg}))

    assert lg.buffer_memory()[1] == lg.buffer_numel()[1] * 4
