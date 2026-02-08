# Copyright (c) AIoWay Authors - All Rights Reserved

import itertools
import logging

from aioway import attrs
from aioway.attrs import Shape

__all__ = ["bcast_same_dim", "can_bcast_dim", "matmul_2d"]

LOGGER = logging.getLogger(__name__)


def permute(shape: Shape, dims: list[int]) -> Shape:
    """
    Permute the original ``shape`` with ``dims``.
    """

    if not shape.valid_dims(dims):
        raise ValueError(f"Invalid dimensions: {dims}")

    if sorted(dims) != list(range(len(shape))):
        raise ValueError(
            f"Dimensions must be the permutation of {list(range(len(shape)))=}."
        )

    wrapped_dims = shape.wrap_dims(dims)

    return attrs.shape(shape[d] for d in wrapped_dims)


def agg(shape: Shape, dims: list[int]) -> Shape:
    """
    Aggregate the target dimensions.
    """

    if not shape.valid_dims(dims):
        raise ValueError(f"Invalid dimensions: {dims}")

    wrapped_dims = shape.wrap_dims(dims)

    # Since ``dims`` is most likely small, not using a ``set`` for ``in``.
    return attrs.shape(s for i, s in enumerate(shape) if i not in wrapped_dims)


def can_bcast_dim(left: int, right: int) -> bool:
    """
    Check if 2 dimensions are comaptible when matched ([n, n], [1, n], [n, 1] allowed)
    """
    return left == right or left == 1 or right == 1


def bcast_same_dim(left: Shape, right: Shape) -> Shape:
    """
    Broadcast the dimensions, where if left[idx] != shape[idx],
    one of them must be 1.
    """

    if len(left) != len(right):
        raise ValueError(f"{len(left)=} != {len(right)=}")

    result: list[int] = [NotImplemented] * len(left)
    for i, l, r in zip(itertools.count(), left, right):
        if not can_bcast_dim(l, r):
            raise ValueError(f"{left[i]=} incompatible with {right[i]=} for {i=}")

        # Since both are natural numbers, the larger one is the one we want.
        result[i] = max(l, r)
    return attrs.shape(result)


def matmul_2d(left: Shape, right: Shape) -> Shape:
    """
    The matmul operator, when both sides are 2D.
    """

    if len(left) != 2 or len(right) != 2:
        raise ValueError(f"{len(left)=}, {len(right)=}, both must be 2.")

    l0, l1 = left
    r0, r1 = right

    if not can_bcast_dim(l1, r0):
        raise ValueError(f"{left[1]=} incompatbile {right[0]}")

    return attrs.shape(l0, r1)
