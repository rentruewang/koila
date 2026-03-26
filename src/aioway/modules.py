# Copyright (c) AIoWay Authors - All Rights Reserved

"The preview module protocol definition."

import abc
import dataclasses as dcls
import functools
from abc import ABC
from collections.abc import Callable, Mapping
from typing import Any

from torch import Tensor
from torch.nn import Module
from torch.nn import Module as NnModule

from aioway._signs import Signature
from aioway._tracking import ModuleApiTracker, logging
from aioway._typing import SeqKeysView
from aioway.attrs import Attr, Device, DeviceLike, DType, DTypeLike, Shape, ShapeLike

__all__ = ["Module"]

LOGGER = logging.get_logger(__name__)


class Module[**P, T: NnModule](Mapping[str, Any], ABC):
    """
    Preview informs us how an `nn.Module` would be behave without initializing it.
    """

    def __init__(self, module: Callable[P, T], *args: P.args, **kwargs: P.kwargs):
        self._module = module
        self._args = args
        self._kwargs = kwargs

    def preview(self, attr: Attr, /) -> Attr:
        """
        Transforms the input attribute into an attribute describing the output.

        Returns `NotImplemented` when the input `Attr` is incompatible (usually device and dtype).
        """

        try:
            return self._preview(attr)
        except ValueError:
            return NotImplemented

    def _preview(self, attr: Attr) -> Attr:
        with self._tracker()(name="attr", signature=Signature(Device, Device)):
            device = self._preview_device(attr.device)

        with self._tracker()(name="attr", signature=Signature(DType, DType)):
            dtype = self._preview_dtype(attr.dtype)

        with self._tracker()(name="attr", signature=Signature(Shape, Shape)):
            shape = self._preview_shape(attr.shape)

        return Attr.parse(device=device, dtype=dtype, shape=shape)

    def _preview_device(self, device: Device) -> DeviceLike:
        "Only allow same device to be ok, or if `self.device is None`."

        if self.device is None:
            return device

        if device == self.device:
            return device

        raise ValueError

    @abc.abstractmethod
    def _preview_dtype(self, dtype: DType) -> DTypeLike:
        """
        Pre-compute the dtype of the output.

        If the input dtype is invalid, `raise ValueError`.
        """

        ...

    @abc.abstractmethod
    def _preview_shape(self, shape: Shape, /) -> ShapeLike:
        """
        Pre-compute the shape for the layer.

        If the input shape cannot be handled, `raise ValueError`
        """

        ...

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Do a forward pass on the input `tensor`.
        """

        with self._tracker()(name="forward", signature=Signature(Tensor, Tensor)):
            return self.module(tensor)

    @classmethod
    def _tracker(cls) -> ModuleApiTracker:
        return ModuleApiTracker(lambda: cls)

    @functools.cached_property
    def module(self) -> Module:
        "The lazy `module` property constructing the `nn.Module`."

        return self.MODULE_TYPE(**self)

    @functools.cached_property
    def __keys_view(self):
        return SeqKeysView([f.name for f in self.__fields])

    @functools.cached_property
    def __fields(self):
        return dcls.fields(self)
