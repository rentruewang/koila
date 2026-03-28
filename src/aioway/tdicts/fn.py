# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing
from abc import ABC
from collections.abc import Iterator
from typing import Any

from tensordict import TensorDict

from aioway import _typing, fake
from aioway.fn import Fn
from aioway.tensors.fn import TensorFn

from .attrs import AttrSet

__all__ = ["TensorDictFn"]


class TensorDictFn(Fn[TensorDict], ABC):
    def __init__(self) -> None:
        super().__init__()
        assert all(fake.is_fake_tensor(tensor) for tensor in self._fake_result.values())
        self.__attrs = AttrSet.from_tensordict(self._fake_result)

    @typing.overload
    def __getitem__(self, key: str) -> TensorFn: ...

    @typing.overload
    def __getitem__(self, key: list[str]) -> TensorDictFn: ...

    @typing.no_type_check
    def __getitem__(self, key):
        if isinstance(key, str):
            from ._selections import GetItemFn

            return GetItemFn(self, key)

        if _typing.is_list_of(str)(key):
            from ._selections import SelectFn

            return SelectFn(self, key)

        raise TypeError(f"Does not handle {type(key)=}.")

    @abc.abstractmethod
    @typing.override
    def _deps(self) -> Iterator[Fn[Any]]:
        raise NotImplementedError

    @property
    def attrs(self):
        return self.__attrs
