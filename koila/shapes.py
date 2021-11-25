from __future__ import annotations

import logging
import math
from abc import abstractmethod
from typing import Any, List, Protocol, Sequence, Tuple, runtime_checkable

from rich.logging import RichHandler

from .errors import UnsupportedError
from .runnables import TensorLike

logger = logging.getLogger(__name__)
logger.addHandler(RichHandler())


@runtime_checkable
class ShapeFunction(Protocol):
    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Tuple[int, ...]:
        ...


def mute_unused_args(*args: Any, **kwargs: Any) -> None:
    del args
    del kwargs


def compatible_dim(input: int, other: int, broadcast: bool = True) -> bool:
    if broadcast:
        return input == 1 or other == 1 or input == other
    else:
        return input == other


def prepends_shape(
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


def coerce_shape(
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

    (input, other) = prepends_shape(input, other, 1)

    shape = []
    for (a, b) in zip(input, other):
        if a <= 0 or b <= 0:
            raise ValueError

        if compatible_dim(a, b):
            shape.append(max(a, b))
        else:
            return None

    return tuple(shape)


def permute_shape(input: Tuple[int, ...], *dims: int) -> Tuple[int, ...]:
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


def tranpose_shape(input: Tuple[int, ...], dim0: int, dim1: int) -> Tuple[int, ...]:
    if len(input) < 2:
        raise ValueError

    shapes = list(input)
    (shapes[dim0], shapes[dim1]) = (shapes[dim1], shapes[dim0])
    return tuple(shapes)


def matmul_shape(input: Tuple[int, ...], other: Tuple[int, ...]) -> Tuple[int, ...]:
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

    (input, other) = prepends_shape(input, other, 1)

    shapes = []
    for (dimi, dimo) in zip(input[:-2], other[:-2]):
        if not compatible_dim(dimi, dimo):
            raise ValueError
        shapes.append(max(dimi, dimo))

    if input[-1] != other[-2]:
        raise ValueError

    shapes.extend([input[-2], other[-1]])

    return tuple(shapes)


def identity(input: TensorLike, *args: Any, **kwargs: Any) -> Tuple[int, ...]:
    mute_unused_args(*args, **kwargs)

    return input.size()


def symmetric(
    input: TensorLike, other: TensorLike, *args: Any, **kwargs: Any
) -> Tuple[int, ...]:
    mute_unused_args(*args, **kwargs)

    shape = coerce_shape(input.size(), other.size(), broadcast=True, scalars=True)

    if shape is None:
        raise ValueError

    return shape


def reduce_dims(
    input: TensorLike,
    dim: int | Tuple[int, ...],
    keepdim: bool = False,
    *args: Any,
    **kwargs: Any,
) -> Tuple[int, ...]:
    mute_unused_args(*args, **kwargs)

    shapes = []

    if isinstance(dim, int):
        dimensions = {dim}
    else:
        dimensions = set(dim)

    for (idx, dimsize) in enumerate(input.size()):
        if idx not in dimensions:
            shapes.append(dimsize)
            continue

        if keepdim:
            shapes.append(1)

    if keepdim:
        assert len(shapes) == input.dim()

    return tuple(shapes)


def scalar(input: TensorLike, *args: Any, **kwargs: Any) -> Tuple[int, ...]:
    mute_unused_args(*args, **kwargs)

    result = reduce_dims(input, dim=tuple(range(input.dim())))
    assert result == ()
    return result


def permute(input: TensorLike, *dims: int, **kwargs: Any) -> Tuple[int, ...]:
    mute_unused_args(**kwargs)

    return permute_shape(input.size(), *dims)


def tranpose(
    input: TensorLike, dim0: int, dim1: int, *args: Any, **kwargs: Any
) -> Tuple[int, ...]:
    mute_unused_args(*args, **kwargs)

    return tranpose_shape(input.size(), dim0, dim1)


def matmul(
    input: TensorLike, other: TensorLike, *args: Any, **kwargs: Any
) -> Tuple[int, ...]:
    mute_unused_args(*args, **kwargs)

    return matmul_shape(input.size(), other.size())


def linear(
    input: TensorLike,
    weight: TensorLike,
    bias: TensorLike | None = None,
    *args: Any,
    **kwargs: Any,
) -> Tuple[int, ...]:
    mute_unused_args(*args, **kwargs)

    result = matmul_shape(input.size(), tranpose_shape(weight.size(), -1, -2))

    if bias is not None:
        result = coerce_shape(result, bias.size())

    if result is None:
        raise ValueError

    return result


def cat(
    tensors: Sequence[TensorLike], dim: int = 0, *args: Any, **kwargs: Any
) -> Tuple[int, ...]:
    mute_unused_args(*args, **kwargs)

    if len(tensors) == 0:
        raise ValueError("Expected a sequence of tensors. Got empty sequence.")

    shapes = [t.size() for t in tensors]
    no_dim = [t[:dim] + t[dim + 1 :] for t in shapes]

    result_size = no_dim[0]
    for size in no_dim[1:]:
        if result_size != size:
            raise ValueError(
                f"Dimension should be equal outside dim {dim}. Got {shapes}."
            )

    concat_size = sum(t[dim] for t in shapes)
    return (*result_size[:dim], concat_size, *result_size[dim:])


def pad(
    input: TensorLike, pad: List[int], *args: Any, **kwargs: Any
) -> Tuple[int, ...]:
    shapes = input.size()

    if len(pad) % 2 == 1:
        raise ValueError(f"Length of pad must be divisible by 2. Got {len(pad)}.")

    if len(pad) > (maxlen := len(shapes) * 2):
        raise ValueError(
            f"Padding is way too long. Got {pad}, but {maxlen} is the maximum dimensions allowed."
        )

    pad = (2 * len(shapes) - len(pad)) * [0] + list(reversed(pad))

    pad0 = pad[0::2]
    pad1 = pad[1::2]

    assert len(pad0) == len(pad1) == len(shapes), [pad0, pad1, shapes]

    return tuple(s + p0 + p1 for (s, p0, p1) in zip(shapes, pad0, pad1))


def _int_to_tuple(value: int | Tuple[int, ...], length: int) -> Tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * length

    assert isinstance(value, Tuple)
    assert len(value) == length
    return value


def conv(
    input: TensorLike,
    weight: TensorLike,
    bias: TensorLike | None = None,
    stride: int | Tuple[int, ...] = 1,
    padding: int | Tuple[int, ...] | str = "valid",
    dilation: int | Tuple[int, ...] = 1,
    groups: int = 1,
    *args: Any,
    **kwargs: Any,
) -> Tuple[int, ...]:
    mute_unused_args(groups, *args, **kwargs)

    (batch, chan, *dims) = input.size()
    (out_chan, in_chan, *kernels) = weight.size()

    assert chan == in_chan

    if bias is not None:
        assert coerce_shape(bias.size(), (out_chan,)) is not None

    if isinstance(padding, str):
        raise UnsupportedError

    stride = _int_to_tuple(stride, len(dims))
    padding = _int_to_tuple(padding, len(dims))
    dilation = _int_to_tuple(dilation, len(dims))

    assert len(dims) == len(kernels) == len(stride) == len(padding) == len(dilation)

    out_dims = [
        math.floor((dim + 2 * pad - dil * (ker - 1) - 1) / st + 1)
        for (dim, pad, dil, ker, st) in zip(dims, padding, dilation, kernels, stride)
    ]

    return (batch, out_chan, *out_dims)


def conv_transpose(
    input: TensorLike,
    weight: TensorLike,
    bias: TensorLike | None = None,
    stride: int | Tuple[int, ...] = 1,
    padding: int | Tuple[int, ...] = 0,
    output_padding: int | Tuple[int, ...] = 0,
    groups: int = 1,
    dilation: int | Tuple[int, ...] = 1,
    *args: Any,
    **kwargs: Any,
) -> Tuple[int, ...]:
    mute_unused_args(groups, *args, **kwargs)

    (batch, chan, *dims) = input.size()
    (in_chan, out_chan, *kernels) = weight.size()

    assert chan == in_chan

    if bias is not None:
        assert coerce_shape(bias.size(), (out_chan,)) is not None

    stride = _int_to_tuple(stride, len(dims))
    padding = _int_to_tuple(padding, len(dims))
    output_padding = _int_to_tuple(output_padding, len(dims))
    dilation = _int_to_tuple(dilation, len(dims))

    assert len(dims) == len(kernels) == len(stride) == len(padding) == len(dilation)

    out_dims = [
        (dim - 1) * st - 2 * pad + dil * (ker - 1) + opad + 1
        for (dim, st, pad, dil, ker, opad) in zip(
            dims, stride, padding, dilation, kernels, output_padding
        )
    ]

    return (batch, out_chan, *out_dims)


def pool(
    input: TensorLike,
    *,
    kernel_size: int | Tuple[int, ...],
    stride: int | Tuple[int, ...] = (),
    padding: int | Tuple[int, ...] = 0,
    dilation: int | Tuple[int, ...] = 1,
    ceil_mode: bool = False,
) -> Tuple[int, ...]:
    (batch, chan, *dims) = input.size()

    kernel_size = _int_to_tuple(kernel_size, len(dims))
    stride = _int_to_tuple(stride, len(dims))
    padding = _int_to_tuple(padding, len(dims))
    dilation = _int_to_tuple(dilation, len(dims))

    rounding = math.ceil if ceil_mode else math.floor
    out_dims = [
        rounding((dim + 2 * pad - dil * (ker - 1) - 1) / st + 1)
        for (dim, pad, dil, ker, st) in zip(
            dims, padding, dilation, kernel_size, stride
        )
    ]

    return (batch, chan, *out_dims)


def maxpool(
    input: TensorLike,
    kernel_size: int | Tuple[int, ...],
    stride: int | Tuple[int, ...] = (),
    padding: int | Tuple[int, ...] = 0,
    dilation: int | Tuple[int, ...] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
    *args: Any,
    **kwargs: Any,
) -> Tuple[int, ...]:
    mute_unused_args(*args, **kwargs)

    if return_indices:
        raise UnsupportedError

    return pool(
        input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


def avgpool(
    input: TensorLike,
    kernel_size: int | Tuple[int, ...],
    stride: int | Tuple[int, ...] = (),
    padding: int | Tuple[int, ...] = 0,
    ceil_mode: bool = False,
    *args: Any,
    **kwargs: Any,
) -> Tuple[int, ...]:
    mute_unused_args(*args, **kwargs)

    return pool(
        input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
    )
