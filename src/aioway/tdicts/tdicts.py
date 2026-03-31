# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing
from abc import ABC
from collections import abc as cabc

import tensordict as td

from aioway import _typing, fake
from aioway.fn import Fn
from aioway.tensors.tensors import TensorFn

from .attrs import AttrSet

__all__ = ["TensorDictFn", "tdict"]


class TensorDictFn(Fn[td.TensorDict], cabc.Mapping[str, TensorFn], ABC):
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
            from . import _selections

            return _selections.GetItemFn(self, key)

        if _typing.is_list_of(str)(key):
            from . import _selections

            return _selections.SelectFn(self, key)

        raise TypeError(f"Does not handle {type(key)=}.")

    @typing.override
    def __len__(self) -> int:
        return len(self.attrs)

    @typing.override
    def __contains__(self, key: object, /) -> bool:
        if isinstance(key, str):
            return key in self.keys()

        return False

    @typing.override
    def __iter__(self):
        yield from self.keys()

    @typing.override
    def keys(self):
        return self.attrs.keys()

    @abc.abstractmethod
    @typing.override
    def _deps(self) -> cabc.Iterator[Fn[typing.Any]]:
        raise NotImplementedError

    @property
    def attrs(self):
        return self.__attrs

    @classmethod
    def from_tensordict(cls, data: td.TensorDict) -> TensorDictFn:
        from . import _data

        return _data.TensorDictDataFn(data)


def tdict(item: TensorDictFn | td.TensorDict) -> TensorDictFn:
    if isinstance(item, TensorDictFn):
        return item

    if isinstance(item, td.TensorDict):
        return TensorDictFn.from_tensordict(item)

    raise TypeError(f"Do not know how to handle {type(item)=}.")
