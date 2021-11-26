from __future__ import annotations

import logging
import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, List, Protocol, Sequence, Tuple, overload, runtime_checkable

from rich.logging import RichHandler
from torch.functional import Tensor

from . import constants, runnables
from .errors import UnsupportedError
from .runnables import MetaData, TensorLike

logger = logging.getLogger(__name__)
logger.addHandler(RichHandler())


@dataclass(init=False, frozen=True)
class Shape:
    value: Tuple[int, ...]
    metadata: MetaData

    def __init__(self, value: Sequence[int], metadata: MetaData) -> None:
        object.__setattr__(self, "value", tuple(value))
        object.__setattr__(self, "metadata", metadata)

    def __iter__(self):
        return iter(self.value)

    @overload
    def __getitem__(self, index: int) -> int:
        ...

    @overload
    def __getitem__(self, index: slice) -> Tuple[int, ...]:
        ...

    def __getitem__(self, index: int | slice) -> int | Tuple[int, ...]:
        return self.value[index]

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Shape):
            return self == other

        if isinstance(other, Tuple):
            return self.value == other

        return False


@runtime_checkable
class ShapeFunction(Protocol):
    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Shape:
        ...


def mute_unused_args(*args: Any, **kwargs: Any) -> None:
    del args
    del kwargs


def same(*tensors: TensorLike) -> MetaData:
    assert len(tensors) > 0
    if len(tensors) == 1:
        runnables.dtype(tensors[0])

    dtypes = [runnables.dtype(t) for t in tensors]

    max_dtype = max(dtypes, key=lambda dtype: constants.MEMORY_BYTES[dtype])

    devices = [str(runnables.device(t)) for t in tensors]

    if len(set(devices)) != 1:
        raise ValueError(f"Expected tensors to be on the same device, got {devices}.")

    return MetaData(max_dtype, devices[0])


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


def identity(input: TensorLike, *args: Any, **kwargs: Any) -> Shape:
    mute_unused_args(*args, **kwargs)

    return Shape(input.size(), same(input))


def symmetric(input: TensorLike, other: TensorLike, *args: Any, **kwargs: Any) -> Shape:
    mute_unused_args(*args, **kwargs)

    shape = coerce_shape(input.size(), other.size(), broadcast=True, scalars=True)

    if shape is None:
        raise ValueError

    return Shape(shape, same(input, other))


def reduce_dims(
    input: TensorLike,
    dim: int | Tuple[int, ...] | None = None,
    keepdim: bool = False,
    *args: Any,
    **kwargs: Any,
) -> Shape:
    mute_unused_args(*args, **kwargs)

    shapes = []

    if dim is None:
        dimensions = set(range(input.dim()))
    elif isinstance(dim, int):
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

    return Shape(shapes, same(input))


def permute(input: TensorLike, *dims: int, **kwargs: Any) -> Shape:
    mute_unused_args(**kwargs)

    return Shape(permute_shape(input.size(), *dims), same(input))


def tranpose(
    input: TensorLike, dim0: int, dim1: int, *args: Any, **kwargs: Any
) -> Shape:
    mute_unused_args(*args, **kwargs)

    return Shape(tranpose_shape(input.size(), dim0, dim1), same(input))


def select(
    input: TensorLike,
    dim: int | ... | None,
    index: int | Tensor,
    *args: Any,
    **kwargs: Any,
) -> Shape:
    mute_unused_args(*args, **kwargs)

    shape = input.size()

    if dim is ...:
        dim = -1

    if dim is None:
        dim = 0
        shape = (1,) + shape

    if not -len(shape) <= dim < len(shape):
        raise IndexError

    dim %= len(shape)
    assert isinstance(dim, int)

    if isinstance(index, Tensor):
        sliced_idx = (len(index),)
    else:
        sliced_idx = ()

    return Shape(shape[:dim] + sliced_idx + shape[dim + 1 :], same(input))


def embedding(
    input: TensorLike, weight: TensorLike, *args: Any, **kwargs: Any
) -> Shape:
    mute_unused_args(*args, **kwargs)

    shape = input.size()
    return Shape((*shape, weight.size(-1)), same(input))


def matmul(input: TensorLike, other: TensorLike, *args: Any, **kwargs: Any) -> Shape:
    mute_unused_args(*args, **kwargs)

    return Shape(matmul_shape(input.size(), other.size()), same(input, other))


def linear(
    input: TensorLike,
    weight: TensorLike,
    bias: TensorLike | None = None,
    *args: Any,
    **kwargs: Any,
) -> Shape:
    mute_unused_args(*args, **kwargs)

    result = matmul_shape(input.size(), tranpose_shape(weight.size(), -1, -2))

    if bias is not None:
        result = coerce_shape(result, bias.size())

    if result is None:
        raise ValueError

    return Shape(result, same(input, weight))


def cat(
    tensors: Sequence[TensorLike], dim: int = 0, *args: Any, **kwargs: Any
) -> Shape:
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
    return Shape([*result_size[:dim], concat_size, *result_size[dim:]], same(*tensors))


def pad(input: TensorLike, pad: List[int], *args: Any, **kwargs: Any) -> Shape:
    mute_unused_args(*args, **kwargs)

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

    return Shape([s + p0 + p1 for (s, p0, p1) in zip(shapes, pad0, pad1)], same(input))


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
) -> Shape:
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

    return Shape((batch, out_chan, *out_dims), same(input, weight))


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
) -> Shape:
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

    return Shape((batch, out_chan, *out_dims), same(input, weight))


def pool(
    input: TensorLike,
    *,
    kernel_size: int | Tuple[int, ...],
    stride: int | Tuple[int, ...] = (),
    padding: int | Tuple[int, ...] = 0,
    dilation: int | Tuple[int, ...] = 1,
    ceil_mode: bool = False,
) -> Shape:
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

    return Shape((batch, chan, *out_dims), same(input))


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
) -> Shape:
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
) -> Shape:
    mute_unused_args(*args, **kwargs)

    return pool(
        input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
    )
