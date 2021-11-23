from __future__ import annotations

from abc import abstractmethod
from typing import Any, Protocol, Tuple


class ShapeFunction(Protocol):
    @abstractmethod
    def __call__(
        self, input: Tuple[int, ...], *args: Any, **kwds: Any
    ) -> Tuple[int, ...]:
        ...


def compatible(a: int, b: int, broadcast: bool = True) -> bool:
    if broadcast:
        return a == 1 or b == 1 or a == b
    else:
        return a == b


def coerce(
    input: Tuple[int, ...],
    other: Tuple[int, ...],
    broadcast: bool = True,
    scalars: bool = True,
) -> Tuple[int, ...] | None:
    if scalars:
        if len(input) == 0:
            return other

        if len(other) == 0:
            return input

    if len(input) != len(other):
        return None

    if not broadcast:
        if (shape := input) == other:
            return shape
        else:
            return None

    shape = []
    for (a, b) in zip(input, other):
        if a <= 0 or b <= 0:
            raise ValueError

        if compatible(a, b):
            shape.append(max(a, b))
        else:
            return None

    return tuple(shape)


def identity(input: Tuple[int, ...], *args: Any, **kwargs: Any) -> Tuple[int, ...]:
    _ = args
    _ = kwargs
    return input


def symmetric(
    input: Tuple[int, ...], other: Tuple[int, ...], *args: Any, **kwargs: Any
) -> Tuple[int, ...]:
    _ = args
    _ = kwargs

    shape = coerce(input, other, broadcast=True, scalars=True)

    if shape is None:
        raise ValueError

    return shape


def scalar(*args: Any, **kwargs: Any) -> Tuple[int, ...]:
    _ = args
    _ = kwargs

    return ()


def reduce_dims(
    input: Tuple[int, ...], dim: int | Tuple[int, ...], *args: Any, **kwargs: Any
) -> Tuple[int, ...]:
    _ = args
    _ = kwargs

    shapes = list(input)

    if isinstance(dim, int):
        dim = (dim,)

    for d in dim:
        shapes[d] = 1

    return tuple(shapes)


def permute(input: Tuple[int, ...], *dims: int, **kwargs: Any) -> Tuple[int, ...]:
    _ = kwargs
    if not len(input) == len(dims):
        raise ValueError

    if sorted(dims) != list(range(len(dims))):
        raise ValueError

    if not len(set(dims)) == len(dims):
        raise ValueError

    dims_order_pair = sorted(enumerate(dims), key=lambda pair: pair[1])
    scattered_dims = [pair[0] for pair in dims_order_pair]
    paired = sorted(zip(scattered_dims, input))
    reordered_dim = [pair[1] for pair in paired]
    return tuple(reordered_dim)
    


def tranpose(
    input: Tuple[int, ...], dim0: int, dim1: int, *args: Any, **kwargs: Any
) -> Tuple[int, ...]:
    _ = args
    _ = kwargs

    if len(input) < 2:
        raise ValueError

    shapes = list(input)
    (shapes[dim0], shapes[dim1]) = (shapes[dim1], shapes[dim0])
    return tuple(shapes)


def matmul(
    input: Tuple[int, ...], other: Tuple[int, ...], *args: Any, **kwargs: Any
) -> Tuple[int, ...]:
    _ = args
    _ = kwargs

    li = len(input)
    lo = len(other)

    if li == 0 or lo == 0:
        raise ValueError(
            "Both arguments to matmul need to be at least 1D."
            " "
            f"Got {li}D and {lo}D."
        )

    if li == lo == 1:
        if input[0] != other[0]:
            raise ValueError

        return ()

    if li == lo == 2:
        if input[1] != other[0]:
            raise ValueError

        return (input[0], other[1])

    if li == 1 and other == 2:
        if input[0] != other[0]:
            raise ValueError

        return (other[1],)

    if li == 2 and other == 1:
        if input[1] != other[0]:
            raise ValueError

        return (input[0],)

    prepend = (1,) * abs(li - lo)
    if li >= lo:
        other = prepend + other
    else:
        input = prepend + input
    assert len(input) == len(other)

    shapes = []
    for (dimi, dimo) in zip(input[:-2], other[:-2]):
        if not compatible(dimi, dimo):
            raise ValueError
        shapes.append(max(dimi, dimo))

    if input[-1] != other[-2]:
        raise ValueError
    shapes.extend([input[-2], other[-1]])

    return tuple(shapes)
