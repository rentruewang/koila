import torch

from koila import PrePassFunc, prepasses

from . import common


def test_compatibility() -> None:
    assert isinstance(prepasses.identity, PrePassFunc)
    assert isinstance(prepasses.symmetric, PrePassFunc)
    assert isinstance(prepasses.reduce_dims, PrePassFunc)
    assert isinstance(prepasses.permute, PrePassFunc)
    assert isinstance(prepasses.tranpose, PrePassFunc)
    assert isinstance(prepasses.matmul, PrePassFunc)
    assert isinstance(prepasses.linear, PrePassFunc)
    assert isinstance(prepasses.cat, PrePassFunc)
    assert isinstance(prepasses.pad, PrePassFunc)
    assert isinstance(prepasses.conv, PrePassFunc)
    assert isinstance(prepasses.conv_transpose, PrePassFunc)
    assert isinstance(prepasses.pool, PrePassFunc)
    assert isinstance(prepasses.maxpool, PrePassFunc)
    assert isinstance(prepasses.avgpool, PrePassFunc)


def test_identity() -> None:
    common.call(
        common.assert_equal,
        [
            [prepasses.identity(torch.randn(1, 2, 3, 4, 5)), (1, 2, 3, 4, 5)],
            [prepasses.identity(torch.randn(4, 2, 5)), (4, 2, 5)],
            [prepasses.identity(torch.randn(17, 1, 4)), (17, 1, 4)],
        ],
    )


def test_symmetric() -> None:
    common.call(
        common.assert_equal,
        [
            [prepasses.symmetric(torch.randn(2, 4, 5), torch.randn(())), (2, 4, 5)],
            [
                prepasses.symmetric(torch.randn(2, 4, 5), torch.randn(2, 4, 5)),
                (2, 4, 5),
            ],
            [
                prepasses.symmetric(torch.randn(2, 1, 5), torch.randn(2, 4, 5)),
                (2, 4, 5),
            ],
            [
                prepasses.symmetric(torch.randn(2, 1, 5), torch.randn(2, 4, 1)),
                (2, 4, 5),
            ],
        ],
    )


def test_reduce_dims() -> None:
    common.call(
        common.assert_equal,
        [
            [prepasses.reduce_dims(torch.randn(1, 2, 3, 4, 5), 1), (1, 3, 4, 5)],
            [prepasses.reduce_dims(torch.randn(1, 2, 3, 4, 5), (2, 3)), (1, 2, 5)],
            [
                prepasses.reduce_dims(torch.randn(5, 2, 3, 4), (2, 3), keepdim=True),
                (5, 2, 1, 1),
            ],
        ],
    )


def test_scalar() -> None:
    common.call(
        common.assert_equal,
        [
            [prepasses.reduce_dims(torch.randn(5, 5, 2)), ()],
            [prepasses.reduce_dims(torch.randn(7, 8)), ()],
        ],
    )


def test_matmul() -> None:
    common.call(
        common.assert_equal,
        [
            [prepasses.matmul(torch.randn(8), torch.randn(8)), ()],
            [prepasses.matmul(torch.randn(8, 3), torch.randn(3)), (8,)],
            [prepasses.matmul(torch.randn(8), torch.randn(8, 3)), (3,)],
            [prepasses.matmul(torch.randn(4, 5), torch.randn(5, 3)), (4, 3)],
            [prepasses.matmul(torch.randn(9, 4, 5), torch.randn(9, 5, 3)), (9, 4, 3)],
            [prepasses.matmul(torch.randn(9, 4, 5), torch.randn(1, 5, 3)), (9, 4, 3)],
            [
                prepasses.matmul(torch.randn(9, 7, 4, 5), torch.randn(1, 5, 3)),
                (9, 7, 4, 3),
            ],
        ],
    )


def test_transpose() -> None:
    common.call(
        common.assert_equal,
        [[prepasses.tranpose(torch.randn(3, 4, 5), 1, 2), (3, 5, 4)]],
    )


def test_linear() -> None:
    common.call(
        common.assert_equal,
        [
            [
                prepasses.linear(
                    torch.randn(7, 11, 13),
                    weight=torch.randn(17, 13),
                    bias=torch.randn(17),
                ),
                (7, 11, 17),
            ]
        ],
    )


def test_cat() -> None:
    common.call(
        common.assert_equal,
        [
            [prepasses.cat([torch.randn(2, 3, 5), torch.randn(3, 3, 5)]), (5, 3, 5)],
            [
                prepasses.cat([torch.randn(2, 3, 5), torch.randn(2, 4, 5)], dim=1),
                (2, 7, 5),
            ],
        ],
    )
