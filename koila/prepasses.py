from __future__ import annotations

import functools
import logging
import math
import operator
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    List,
    Literal,
    Protocol,
    Sequence,
    Tuple,
    overload,
    runtime_checkable,
)

from rich.logging import RichHandler
from torch import device as Device
from torch import dtype as DType
from torch.functional import Tensor

from . import constants, runnables, shapes
from .errors import UnsupportedError
from .runnables import BatchInfo
from .tensors import TensorLike

logger = logging.getLogger(__name__)
logger.addHandler(RichHandler())


class CallBack(Protocol):
    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Reducer:
        ...


class Reducer(Protocol):
    @abstractmethod
    def __call__(self, result: Tensor, /) -> Tensor:
        ...


@dataclass(frozen=True)
class MetaData:
    dtype: DType
    device: str | Device
    batch: BatchInfo | None
    reducer: CallBack | None


@dataclass
class PrePass:
    shape: Tuple[int, ...]
    metadata: MetaData

    def __init__(self, shape: Sequence[int], metadata: MetaData) -> None:
        self.shape = tuple(shape)
        self.metadata = metadata

    def __iter__(self):
        return iter(self.shape)

    @overload
    def __getitem__(self, index: int) -> int:
        ...

    @overload
    def __getitem__(self, index: slice) -> Tuple[int, ...]:
        ...

    def __getitem__(self, index: int | slice) -> int | Tuple[int, ...]:
        return self.shape[index]

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, PrePass):
            return self == other

        if isinstance(other, Tuple):
            return self.shape == other

        return False

    @property
    def dtype(self) -> DType:
        return self.metadata.dtype

    @property
    def device(self) -> str | Device:
        return self.metadata.device

    def batch(self) -> BatchInfo | None:
        return self.metadata.batch

    def reducer(self) -> CallBack | None:
        return self.metadata.reducer


@runtime_checkable
class PrePassFunc(Protocol):
    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> PrePass:
        ...


def mute_unused_args(*args: Any, **kwargs: Any) -> None:
    del args
    del kwargs


def trivial(input: Tensor, *args: Any, **kwargs: Any) -> Reducer:
    mute_unused_args(input, *args, **kwargs)
    return lambda result: result


def same(
    tensors: Sequence[TensorLike], batch: BatchInfo | None, reducer: CallBack | None
) -> MetaData:
    assert len(tensors) > 0
    dtypes = [t.dtype for t in tensors]

    max_dtype = max(dtypes, key=lambda typ: constants.MEMORY_BYTES[typ])

    devices = [str(t.device) for t in tensors]

    if len(set(devices)) != 1:
        raise ValueError(f"Expected tensors to be on the same device, got {devices}.")

    return MetaData(max_dtype, devices[0], batch, reducer)


def identity(input: TensorLike, /, *args: Any, **kwargs: Any) -> PrePass:
    mute_unused_args(*args, **kwargs)

    return PrePass(input.size(), same([input], input.batch(), trivial))


def symmetric(
    input: TensorLike, other: TensorLike, /, *args: Any, **kwargs: Any
) -> PrePass:
    mute_unused_args(*args, **kwargs)

    shape = shapes.coerce(input.size(), other.size(), broadcast=True, scalars=True)

    if shape is None:
        raise ValueError

    batch = None
    if (b := input.batch()) == other.batch():
        batch = b

    return PrePass(shape, same([input, other], batch, trivial))


def reduce_dims(
    input: TensorLike,
    /,
    dim: int | Tuple[int, ...] | None = None,
    keepdim: bool = False,
    *args: Any,
    **kwargs: Any,
) -> PrePass:
    mute_unused_args(*args, **kwargs)

    (shape, dimensions) = shapes.reduce_dims(input.size(), dim, keepdim)

    if input.batch() in dimensions:
        batch = None
        reducer = None
    else:
        batch = input.batch()
        reducer = trivial

    return PrePass(shape, same([input], batch, reducer))


def scalars(input: TensorLike, /, *args: Any, **kwargs: Any) -> PrePass:
    mute_unused_args(*args, **kwargs)

    return reduce_dims(input, tuple(range(input.dim())))


def mean(
    input: TensorLike,
    /,
    dim: int | Tuple[int, ...] | None = None,
    keepdim: bool = False,
    *args: Any,
    **kwargs: Any,
) -> PrePass:
    mute_unused_args(*args, **kwargs)

    (shape, dimensions) = shapes.reduce_dims(input.size(), dim, keepdim)

    if (b := input.batch()) in dimensions:
        batch = None

        def mean_callback(input: Tensor, *args: Any, **kwargs: Any) -> Reducer:
            def reducer(result: Tensor) -> Tensor:
                return result * input.size(b) / shape[b]

            return reducer

        reducer = mean_callback
    else:
        batch = input.batch()
        reducer = trivial
    return PrePass(shape, same([input], batch, reducer))


def permute(input: TensorLike, /, *dims: int, **kwargs: Any) -> PrePass:
    mute_unused_args(**kwargs)

    mapping = dict(enumerate(dims))

    batch = None
    if (b := input.batch()) is not None:
        batch = b.map(lambda x: mapping[x])

    return PrePass(shapes.permute(input.size(), *dims), same([input], batch, trivial))


def reshape(input: TensorLike, /, *shape: int, **kwargs: Any) -> PrePass:
    mute_unused_args(**kwargs)

    shape = shapes.reshape(input.size(), *shape)

    batch = None
    if (b := input.batch()) is not None:
        if b in shape:
            batch = b.map(shape.index)

    return PrePass(shape, same([input], batch, trivial))


def view(input: TensorLike, /, *shape: int, **kwargs: Any) -> PrePass:
    mute_unused_args(**kwargs)

    shape = shapes.view(input.size(), *shape)

    batch = None
    if (b := input.batch()) is not None:
        if b in shape:
            batch = b.map(shape.index)

    return PrePass(shape, same([input], batch, trivial))


def flatten(
    input: TensorLike,
    /,
    start_dim: int = 0,
    end_dim: int = -1,
    *args: Any,
    **kwargs: Any,
) -> PrePass:
    logger.debug("%s, %s, %s", input.size(), start_dim, end_dim)

    mute_unused_args(*args, **kwargs)

    start_dim %= input.dim()
    end_dim %= input.dim()

    sizes = input.size()

    shape = (
        *sizes[:start_dim],
        functools.reduce(operator.mul, sizes[start_dim : end_dim + 1]),
        *sizes[end_dim + 1 :],
    )

    batch = None
    if (b := input.batch()) is not None:
        if not (start_dim <= b.index <= end_dim):
            batch = b

    return PrePass(shape, same([input], batch, trivial))


def tranpose(
    input: TensorLike, dim0: int, dim1: int, /, *args: Any, **kwargs: Any
) -> PrePass:
    mute_unused_args(*args, **kwargs)

    batch = None
    if (b := input.batch()) is not None:
        batch = b.map(lambda x: {dim0: dim1, dim1: dim0}[x])

    return PrePass(
        shapes.tranpose(input.size(), dim0, dim1), same([input], batch, trivial)
    )


def select(
    input: TensorLike,
    dim: int | ... | None,
    index: int | Tensor,
    /,
    *args: Any,
    **kwargs: Any,
) -> PrePass:
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

    batch = None
    if (b := input.batch()) != dim:
        batch = b

    return PrePass(
        shape[:dim] + sliced_idx + shape[dim + 1 :],
        same([input], batch, trivial),
    )


def embedding(
    input: TensorLike, weight: TensorLike, /, *args: Any, **kwargs: Any
) -> PrePass:
    mute_unused_args(*args, **kwargs)

    shape = input.size()
    return PrePass(
        (*shape, weight.size(-1)),
        same([input], input.batch(), trivial),
    )


def matmul(
    input: TensorLike, other: TensorLike, /, *args: Any, **kwargs: Any
) -> PrePass:
    mute_unused_args(*args, **kwargs)

    if (batch := input.batch()) != other.batch():
        raise UnsupportedError

    return PrePass(
        shapes.matmul(input.size(), other.size()),
        same([input, other], input.batch(), trivial),
    )


def loss(
    input: TensorLike,
    target: TensorLike,
    /,
    reduction: Literal["none", "mean", "sum"] = "mean",
    *args: Any,
    **kwargs: Any,
) -> PrePass:
    mute_unused_args(*args, **kwargs)

    # Currently only supports tensors of the same batch size.
    if (batch := input.batch()) != target.batch():
        raise UnsupportedError

    output_shape = {
        "none": input.size(),
        "mean": (),
        "sum": (),
    }[reduction]

    reducer = {"none": trivial, "mean": trivial, "sum": trivial}[reduction]

    return PrePass(output_shape, same([input, target], batch, reducer))


def linear(
    input: TensorLike,
    weight: TensorLike,
    bias: TensorLike | None = None,
    *args: Any,
    **kwargs: Any,
) -> PrePass:
    mute_unused_args(*args, **kwargs)

    result = shapes.matmul(input.size(), shapes.tranpose(weight.size(), -1, -2))

    if bias is not None:
        result = shapes.coerce(result, bias.size())

    if result is None:
        raise ValueError

    return PrePass(result, same([input, weight], input.batch(), trivial))


def cat(
    tensors: Sequence[TensorLike], dim: int = 0, *args: Any, **kwargs: Any
) -> PrePass:
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

    if len(set(t.batch() for t in tensors)) != 1:
        raise UnsupportedError

    batch = None
    if (b := tensors[0].batch()) != dim:
        batch = b

    concat_size = sum(t[dim] for t in shapes)
    return PrePass(
        [*result_size[:dim], concat_size, *result_size[dim:]],
        same(tensors, batch, trivial),
    )


def pad(input: TensorLike, pad: List[int], *args: Any, **kwargs: Any) -> PrePass:
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

    return PrePass(
        [s + p0 + p1 for (s, p0, p1) in zip(shapes, pad0, pad1)],
        same([input], input.batch(), trivial),
    )


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
) -> PrePass:
    mute_unused_args(groups, *args, **kwargs)

    (batch, chan, *dims) = input.size()
    (out_chan, in_chan, *kernels) = weight.size()

    assert chan == in_chan

    if bias is not None:
        assert shapes.coerce(bias.size(), (out_chan,)) is not None

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

    return PrePass(
        (batch, out_chan, *out_dims),
        same([input, weight], input.batch(), trivial),
    )


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
) -> PrePass:
    mute_unused_args(groups, *args, **kwargs)

    (batch, chan, *dims) = input.size()
    (in_chan, out_chan, *kernels) = weight.size()

    assert chan == in_chan

    if bias is not None:
        assert shapes.coerce(bias.size(), (out_chan,)) is not None

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

    return PrePass(
        (batch, out_chan, *out_dims),
        same([input, weight], input.batch(), trivial),
    )


def pool(
    input: TensorLike,
    *,
    kernel_size: int | Tuple[int, ...],
    stride: int | Tuple[int, ...] = (),
    padding: int | Tuple[int, ...] = 0,
    dilation: int | Tuple[int, ...] = 1,
    ceil_mode: bool = False,
) -> PrePass:
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

    return PrePass(
        (batch, chan, *out_dims), same([input], input.batch(), trivial)
    )


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
) -> PrePass:
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
) -> PrePass:
    mute_unused_args(*args, **kwargs)

    return pool(
        input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
    )
