# Copyright (c) AIoWay Authors - All Rights Reserved

"Dense layers and convolution layers from `torch.nn`."

import dataclasses as dcls
import math
import typing
from abc import ABC
from typing import ClassVar

from aioway._tracking import logging
from aioway.attrs import DType, DTypeLike, Shape, ShapeLike

from .previews import Preview

__all__ = ["Linear", "Conv1d", "Conv2d", "Conv3d", "Identity"]

LOGGER = logging.get_logger(__name__)


@dcls.dataclass(frozen=True)
class _TransformPreview(Preview, ABC):
    @typing.override
    def _preview_dtype(self, dtype: DType) -> DTypeLike:
        if self.dtype is None:
            return dtype

        return dtype.broadcast(self.dtype)


@dcls.dataclass(frozen=True)
class Linear(_TransformPreview):
    """
    The wrapper for `torch.nn.Linear`.
    """

    from torch.nn import Linear as _Linear

    MODULE_TYPE = _Linear

    in_features: int
    out_features: int
    bias: bool = True

    @typing.override
    def _preview_shape(self, shape: Shape, /) -> ShapeLike:
        if len(shape) <= 1:
            raise ValueError

        if shape[-1] != self.in_features:
            raise ValueError

        return [*shape[:-1], self.out_features]


type ConvSize = int | tuple[int, ...]
"""
Sizing used in convolution is either `int` (applies to all dimensions),
or a `tuple[int, ...]` allowing per-dimension configuration.
"""


@dcls.dataclass(frozen=True)
class _ConvNd(_TransformPreview, ABC):
    in_channels: int
    out_channels: int
    kernel_size: ConvSize
    stride: ConvSize = 1
    padding: ConvSize = 0
    dilation: ConvSize = 1
    bias: bool = True
    padding_mode: str = "zeros"

    NDIM: ClassVar[int]

    def __post_init__(self) -> None:
        """
        Validate `ConvSize` to have `dims` as its length.
        """

        def _check_maybe_tuple(t: ConvSize, name: str):
            if not isinstance(t, tuple):
                return

            if len(t) != self.NDIM:
                raise AssertionError(f"self.{name}={t}, expected ndim={self.NDIM}.")

        _check_maybe_tuple(self.kernel_size, "kernel_size")
        _check_maybe_tuple(self.stride, "stride")
        _check_maybe_tuple(self.padding, "padding")
        _check_maybe_tuple(self.dilation, "dilation")

    @typing.override
    @typing.final
    def _preview_shape(self, shape: Shape, /) -> ShapeLike:
        if len(shape) != self.NDIM + 2:
            raise ValueError

        batch, channels, *dim_ins = shape

        # Should match `in_channels` like `Linear`.
        if channels != self.in_channels:
            raise ValueError

        dim_outs = [self._dim_size(i, dim_in) for i, dim_in in enumerate(dim_ins)]

        return [batch, self.out_channels, *dim_outs]

    def _dim_size(self, at: int, in_size: int) -> int:
        if not 0 <= at < self.NDIM:
            raise ValueError

        return _ConvDim(
            padding=_index_conv_dim(self.padding, dim=at),
            stride=_index_conv_dim(self.stride, dim=at),
            dilation=_index_conv_dim(self.dilation, dim=at),
            kernel_size=_index_conv_dim(self.kernel_size, dim=at),
        )(in_size)


def _index_conv_dim(item: ConvSize, dim: int) -> int:
    match item:
        case int():
            return item
        case tuple():
            return item[dim]


@dcls.dataclass(frozen=True)
class _ConvDim:
    padding: int
    dilation: int
    kernel_size: int
    stride: int

    def __call__(self, in_dim: int) -> int:
        denominator = (
            in_dim + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1
        )
        return math.floor(denominator / self.stride + 1)


@dcls.dataclass(frozen=True)
class Conv1d(_ConvNd, Preview):
    """
    The wrapper for `torch.nn.Conv1d`.
    """

    from torch.nn import Conv1d as _Conv1d

    MODULE_TYPE = _Conv1d
    NDIM = 1


@dcls.dataclass(frozen=True)
class Conv2d(_ConvNd, Preview):
    """
    The wrapper for `torch.nn.Conv2d`.
    """

    from torch.nn import Conv2d as _Conv2d

    MODULE_TYPE = _Conv2d
    NDIM = 2


@dcls.dataclass(frozen=True)
class Conv3d(_ConvNd, Preview):
    """
    The wrapper for `torch.nn.Conv3d`.
    """

    from torch.nn import Conv3d as _Conv3d

    MODULE_TYPE = _Conv3d
    NDIM = 3


@dcls.dataclass(frozen=True)
class Identity(_TransformPreview):
    """
    The wrapper for `torch.nn.Identity`.
    """

    from torch.nn import Identity as _Identity

    MODULE_TYPE = _Identity

    @typing.override
    def _preview_shape(self, shape: Shape, /) -> ShapeLike:
        return shape
