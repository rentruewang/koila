import torch

from koila import ShapeFunction, shapes

from . import common


def test_compatibility() -> None:
    assert isinstance(shapes.identity, ShapeFunction)
    assert isinstance(shapes.symmetric, ShapeFunction)
    assert isinstance(shapes.reduce_dims, ShapeFunction)
    assert isinstance(shapes.scalar, ShapeFunction)
    assert isinstance(shapes.permute, ShapeFunction)
    assert isinstance(shapes.tranpose, ShapeFunction)
    assert isinstance(shapes.matmul, ShapeFunction)
    assert isinstance(shapes.linear, ShapeFunction)
    assert isinstance(shapes.cat, ShapeFunction)
    assert isinstance(shapes.pad, ShapeFunction)


def test_identity() -> None:
    common.call(
        common.assert_equal,
        [
            [shapes.identity(torch.randn(1, 2, 3, 4, 5)), (1, 2, 3, 4, 5)],
            [shapes.identity(torch.randn(4, 2, 5)), (4, 2, 5)],
            [shapes.identity(torch.randn(17, 1, 4)), (17, 1, 4)],
        ],
    )


def test_symmetric() -> None:
    common.call(
        common.assert_equal,
        [
            [shapes.symmetric(torch.randn(2, 4, 5), torch.randn(())), (2, 4, 5)],
            [shapes.symmetric(torch.randn(2, 4, 5), torch.randn(2, 4, 5)), (2, 4, 5)],
            [shapes.symmetric(torch.randn(2, 1, 5), torch.randn(2, 4, 5)), (2, 4, 5)],
            [shapes.symmetric(torch.randn(2, 1, 5), torch.randn(2, 4, 1)), (2, 4, 5)],
        ],
    )


def test_reduce_dims() -> None:
    common.call(
        common.assert_equal,
        [
            [shapes.reduce_dims(torch.randn(1, 2, 3, 4, 5), 1), (1, 3, 4, 5)],
            [shapes.reduce_dims(torch.randn(1, 2, 3, 4, 5), (2, 3)), (1, 2, 5)],
            [
                shapes.reduce_dims(torch.randn(5, 2, 3, 4), (2, 3), keepdim=True),
                (5, 2, 1, 1),
            ],
        ],
    )


def test_scalar() -> None:
    common.call(
        common.assert_equal,
        [
            [shapes.scalar(torch.randn(5, 5, 2)), ()],
            [shapes.scalar(torch.randn(7, 8)), ()],
        ],
    )


def test_matmul() -> None:
    common.call(
        common.assert_equal,
        [
            [shapes.matmul(torch.randn(8), torch.randn(8)), ()],
            [shapes.matmul(torch.randn(8, 3), torch.randn(3)), (8,)],
            [shapes.matmul(torch.randn(8), torch.randn(8, 3)), (3,)],
            [shapes.matmul(torch.randn(4, 5), torch.randn(5, 3)), (4, 3)],
            [shapes.matmul(torch.randn(9, 4, 5), torch.randn(9, 5, 3)), (9, 4, 3)],
            [shapes.matmul(torch.randn(9, 4, 5), torch.randn(1, 5, 3)), (9, 4, 3)],
            [
                shapes.matmul(torch.randn(9, 7, 4, 5), torch.randn(1, 5, 3)),
                (9, 7, 4, 3),
            ],
        ],
    )


def test_transpose() -> None:
    common.call(
        common.assert_equal, [[shapes.tranpose(torch.randn(3, 4, 5), 1, 2), (3, 5, 4)]]
    )


def test_linear() -> None:
    common.call(
        common.assert_equal,
        [
            [
                shapes.linear(
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
            [shapes.cat([torch.randn(2, 3, 5), torch.randn(3, 3, 5)]), (5, 3, 5)],
            [
                shapes.cat([torch.randn(2, 3, 5), torch.randn(2, 4, 5)], dim=1),
                (2, 7, 5),
            ],
        ],
    )
