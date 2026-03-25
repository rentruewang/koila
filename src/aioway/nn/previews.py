# Copyright (c) AIoWay Authors - All Rights Reserved

"The preview module protocol definition."

import abc
import dataclasses as dcls
import functools
import typing
from abc import ABC
from collections.abc import Iterator, KeysView, Mapping
from typing import Any, ClassVar

from torch import Tensor
from torch import device as TorchDevice
from torch import dtype as TorchDType
from torch.nn import Module

from aioway._ops import OpSign
from aioway._tracking import ModuleApiTracker, logging
from aioway._typing import SeqKeysView
from aioway.attrs import Attr, Device, DType, Shape, ShapeLike

__all__ = ["Preview"]

LOGGER = logging.get_logger(__name__)


@dcls.dataclass(frozen=True)
class Preview(Mapping[str, Any], ABC):
    """
    Preview informs us how an `nn.Module` would be behave without initializing it.
    """

    MODULE_TYPE: ClassVar[type[Module]]
    """
    The constructor for the module.
    """

    device: TorchDevice
    dtype: TorchDType

    def __post_init__(self) -> None: ...

    @typing.override
    def keys(self) -> KeysView[str]:
        return self.__keys_view

    @typing.override
    def __len__(self) -> int:
        return len(self.__fields)

    @typing.override
    def __iter__(self) -> Iterator[str]:
        yield from self.keys()

    @typing.override
    def __getitem__(self, key: str) -> Any:
        if key in self.keys():
            return getattr(self, key)

        raise KeyError(key)

    def preview(self, attr: Attr) -> Attr:
        """
        Transforms the input attribute into an attribute describing the output.

        Returns `NotImplemented` when the input `Attr` is incompatible (usually device and dtype).
        """

        with self._tracker()(name="attr", signature=OpSign(Device, Device)):
            if attr.device != self.device:
                return NotImplemented

        with self._tracker()(name="attr", signature=OpSign(DType, DType)):
            dtype = attr.dtype.term * self.dtype

        with self._tracker()(name="attr", signature=OpSign(Shape, Shape)):
            if (shape := self._preview_shape(attr.shape)) is NotImplemented:
                return NotImplemented

        return Attr.parse(
            device=attr.device,
            dtype=dtype.unpack(),
            shape=shape,
        )

    @abc.abstractmethod
    def _preview_shape(self, shape: Shape, /) -> ShapeLike: ...

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Do a forward pass on the input `tensor`.
        """

        with self._tracker()(name="forward", signature=OpSign(Tensor, Tensor)):
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
