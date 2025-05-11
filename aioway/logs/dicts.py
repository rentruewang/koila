# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
import functools
import json
import typing
from abc import ABC
from collections.abc import Callable, Sequence
from typing import Any

from aioway.errors import AiowayError

__all__ = ["AsDict", "GetAttrAsDict", "GetItemAsDict"]


@dcls.dataclass(frozen=True)
class AsDict(ABC):
    """
    Converting a custom object and relevant attributes to a dictionary for logging.
    """

    obj: Any
    """
    Any custom object, whose attribute would be accessed by us.
    """

    attrs: Sequence[str]
    """
    The list of attributes to serialize.
    """

    def __post_init__(self) -> None:
        if not callable(self._get):
            raise AsDictGetterError("The getter supplied is not callable.")

    def __str__(self) -> str:
        return json.dumps({attr: self._get(self.obj, attr) for attr in self.attrs})

    @functools.cached_property
    @abc.abstractmethod
    def _get(self) -> Callable[[Any, str], Any]: ...


@dcls.dataclass(frozen=True)
class GetAttrAsDict(AsDict):
    @property
    @typing.override
    def _get(self):
        return getattr


@dcls.dataclass(frozen=True)
class GetItemAsDict(AsDict):
    @property
    @typing.override
    def _get(self):
        return lambda obj, key: obj[key]


class AsDictGetterError(AiowayError, TypeError, AssertionError): ...
