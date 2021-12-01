from __future__ import annotations

import functools
import logging
import operator
from typing import Set, Tuple

from rich.logging import RichHandler

logger = logging.getLogger(__name__)
logger.addHandler(RichHandler())


def compatible_dim(input: int, other: int, broadcast: bool = True) -> bool:
    if broadcast:
        return input == 1 or other == 1 or input == other
    else:
        return input == other


def prepends(
    input: Tuple[int, ...], other: Tuple[int, ...], value: int
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    logger.debug("Prepending %s and %s.", input, other)

    prepended = (value,) * abs(len(input) - len(other))
    if len(input) >= len(other):
        other = prepended + other
    else:
        input = prepended + input
    assert len(input) == len(other)
    return (input, other)


def coerce(
    input: Tuple[int, ...],
    other: Tuple[int, ...],
    broadcast: bool = True,
    scalars: bool = True,
) -> Tuple[int, ...] | None:
    logger.debug(
        "Coercing %s and %s. Broadcasting: %s. Allow scalars: %s.",
        input,
        other,
        broadcast,
        scalars,
    )

    if scalars:
        if len(input) == 0:
            return other

        if len(other) == 0:
            return input

    if not broadcast:
        if (shape := input) == other:
            return shape
        else:
            return None

    (input, other) = prepends(input, other, 1)

    shape = []
    for (a, b) in zip(input, other):
        if a <= 0 or b <= 0:
            raise ValueError

        if compatible_dim(a, b):
            shape.append(max(a, b))
        else:
            return None

    return tuple(shape)


def permute(input: Tuple[int, ...], *dims: int) -> Tuple[int, ...]:
    logger.debug("%s, %s", input, dims)

    if not len(input) == len(dims):
        raise TypeError

    if sorted(dims) != list(range(len(input))):
        raise ValueError

    if not len(set(dims)) == len(input):
        raise ValueError

    dims_order_pair = sorted(enumerate(dims), key=lambda pair: pair[1])
    scattered_dims = [pair[0] for pair in dims_order_pair]
    paired = sorted(zip(scattered_dims, input))
    reordered_dim = [pair[1] for pair in paired]
    return tuple(reordered_dim)


def reshape(input: Tuple[int, ...], *shape: int) -> Tuple[int, ...]:
    logger.debug("%s, %s", input, shape)

    if not functools.reduce(operator.mul, input) == functools.reduce(
        operator.mul, shape
    ):
        raise ValueError
    return shape


def view(input: Tuple[int, ...], *shape: int) -> Tuple[int, ...]:
    logger.debug("%s, %s", input, shape)

    special_values = [x for x in shape if x < 0]

    if len(special_values) > 1:
        raise ValueError

    if set(special_values) | {-1} != {-1}:
        raise ValueError

    special = -(
        functools.reduce(operator.mul, input) // functools.reduce(operator.mul, shape)
    )
    new_shape = []
    for s in shape:
        if s > 0:
            new_shape.append(s)
        else:
            new_shape.append(special)

    return reshape(input, *new_shape)


def tranpose(input: Tuple[int, ...], dim0: int, dim1: int) -> Tuple[int, ...]:
    logger.debug("%s, %d, %d", input, dim0, dim1)

    if len(input) < 2:
        raise ValueError

    shapes = list(input)
    (shapes[dim0], shapes[dim1]) = (shapes[dim1], shapes[dim0])
    return tuple(shapes)


def matmul(input: Tuple[int, ...], other: Tuple[int, ...]) -> Tuple[int, ...]:
    logger.debug("%s, %s", input, other)

    if len(input) == 0 or len(other) == 0:
        raise ValueError(
            "Both arguments to matmul need to be at least 1D."
            " "
            f"Got {len(input)}D and {len(other)}D."
        )

    if len(input) == len(other) == 1:
        if input[0] != other[0]:
            raise ValueError

        return ()

    if len(input) == len(other) == 2:
        if input[1] != other[0]:
            raise ValueError

        return (input[0], other[1])

    if len(input) == 1 and len(other) == 2:
        if input[0] != other[0]:
            raise ValueError

        return (other[1],)

    if len(input) == 2 and len(other) == 1:
        if input[1] != other[0]:
            raise ValueError

        return (input[0],)

    (input, other) = prepends(input, other, 1)

    shapes = []
    for (dimi, dimo) in zip(input[:-2], other[:-2]):
        if not compatible_dim(dimi, dimo):
            raise ValueError
        shapes.append(max(dimi, dimo))

    if input[-1] != other[-2]:
        raise ValueError

    shapes.extend([input[-2], other[-1]])

    return tuple(shapes)


def reduce_dims(
    input: Tuple[int, ...],
    dim: int | Tuple[int, ...] | None = None,
    keepdim: bool = False,
) -> Tuple[Tuple[int, ...], Set[int]]:
    logger.debug("%s, %s", input, dim)

    shapes = []

    if dim is None:
        dimensions = set(range(len(input)))
    elif isinstance(dim, int):
        dimensions = {dim}
    else:
        dimensions = set(dim)

    for (idx, dimsize) in enumerate(input):
        if idx not in dimensions:
            shapes.append(dimsize)
            continue

        if keepdim:
            shapes.append(1)

    if keepdim:
        assert len(shapes) == len(input)

    return (tuple(shapes), dimensions)
