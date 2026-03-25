# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import functools
import typing
from abc import ABC
from collections.abc import Callable, KeysView, Mapping
from typing import Any

from torch import Tensor
from torch.nn import Module

from aioway._tracking import logging
from aioway._typing import SeqKeysView
from aioway.attrs import Attr

__all__ = ["Preview"]

LOGGER = logging.get_logger(__name__)


@dcls.dataclass(frozen=True)
class Config(Mapping[str, Any]):
    @typing.override
    def keys(self) -> KeysView[str]:
        return self.__keys_view

    @typing.override
    def __len__(self) -> int:
        return len(self.__fields)

    @typing.override
    def __getitem__(self, key: str) -> Any:
        if key in self.keys():
            return getattr(self, key)

        raise KeyError(key)

    @functools.cached_property
    def __keys_view(self):
        return SeqKeysView([f.name for f in self.__fields])

    @functools.cached_property
    def __fields(self):
        return dcls.fields(self)


@dcls.dataclass(frozen=True)
class Preview(ABC):
    """
    Preview lets us know about the layers before we initialize it.
    """

    config: type[Config]
    """
    The configuration type for the module. This is star-unpacked to the constructor of module.
    """

    module: type[Module]
    """
    The constructor for the module.
    """

    attr: Callable[[Config, Attr], Attr]
    """
    Get the output attribute from the input attribute.
    """

    compute: Callable[[Tensor], Tensor]
