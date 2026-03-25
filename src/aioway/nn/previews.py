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
from torch.nn import Module

from aioway._ops import OpSign
from aioway._tracking import ModuleApiTracker, logging
from aioway._typing import SeqKeysView
from aioway.attrs import Attr

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
        """

        with self._tracker()(name="attr", signature=OpSign(Attr, Attr)):
            return self._preview(attr)

    @abc.abstractmethod
    def _preview(self, attr: Attr, /) -> Attr: ...

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
