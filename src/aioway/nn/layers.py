# Copyright (c) AIoWay Authors - All Rights Reserved

"The dense layers from `torch.nn`."

import dataclasses as dcls
import typing

from torch.nn import Linear as _Linear

from aioway._tracking import logging
from aioway.attrs import Shape, ShapeLike

from .previews import Preview

__all__ = ["Linear", "Conv1d", "Conv2d", "Conv3d", "Transformer", "Identity"]

LOGGER = logging.get_logger(__name__)


@dcls.dataclass(frozen=True)
class Linear(Preview):
    """
    The wrapper for `torch.nn.Linear`.
    """

    MODULE_TYPE = _Linear

    in_features: int
    out_features: int
    bias: bool = True

    @typing.override
    def _preview_shape(self, shape: Shape, /) -> ShapeLike:
        if len(shape) <= 1:
            return NotImplemented

        return [*shape[:-1], self.out_features]


type ConvSize = int | tuple[int, ...]
"""
Sizing used in convolution is either `int` (applies to all dimensions),
or a `tuple[int, ...]` allowing per-dimension configuration.
"""


@dcls.dataclass(frozen=True)
class _ConvNd:

    in_channels: int
    out_channels: int
    kernel_size: ConvSize
    stride: ConvSize = 1
    padding: ConvSize = 0
    dilation: ConvSize = 1
    bias: bool = True
    padding_mode: str = "zeros"

    def _validate_sizing(self, dims: int) -> None:
        """
        Validate `ConvSize` to have `dims` as its length.
        """

        def _check_tuple(t: ConvSize, name: str):
            if not isinstance(t, tuple):
                return

            if len(t) != dims:
                raise ValueError(f"self.{name}={t}, expected ndim={dims}.")

        _check_tuple(self.kernel_size, "kernel_size")
        _check_tuple(self.stride, "stride")
        _check_tuple(self.padding, "padding")
        _check_tuple(self.dilation, "dilation")


@dcls.dataclass(frozen=True)
class Conv1d(_ConvNd, Preview):
    """
    The wrapper for `torch.nn.Conv1d`.
    """

    def __post_init__(self) -> None:
        self._validate_sizing(1)

    @typing.override
    def _preview_shape(self, shape: Shape, /) -> ShapeLike:
        raise NotImplementedError


@dcls.dataclass(frozen=True)
class Conv2d(_ConvNd, Preview):
    """
    The wrapper for `torch.nn.Conv2d`.
    """

    def __post_init__(self) -> None:
        self._validate_sizing(2)

    @typing.override
    def _preview_shape(self, shape: Shape, /) -> ShapeLike:
        raise NotImplementedError


@dcls.dataclass(frozen=True)
class Conv3d(_ConvNd, Preview):
    """
    The wrapper for `torch.nn.Conv3d`.
    """

    def __post_init__(self) -> None:
        self._validate_sizing(3)

    @typing.override
    def _preview_shape(self, shape: Shape, /) -> ShapeLike:
        raise NotImplementedError


@dcls.dataclass(frozen=True)
class Transformer(Preview):
    """
    The wrapper for `torch.nn.Transformer`.
    """

    @typing.override
    def _preview_shape(self, shape: Shape, /) -> ShapeLike:
        raise NotImplementedError


@dcls.dataclass(frozen=True)
class Identity(Preview):
    """
    The wrapper for `torch.nn.Identity`.

    This has no arguments.
    """

    @typing.override
    def _preview_shape(self, shape: Shape, /) -> ShapeLike:
        return shape
