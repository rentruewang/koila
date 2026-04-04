# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing
from collections import abc as cabc

import numpy as np
import tensordict as td
import torch

from aioway import _common, _typing, fake, fn, meta

from . import tensors

__all__ = ["TensorDictFn", "tdict"]


class TensorDictFn(fn.Fn[td.TensorDict], cabc.Mapping[str, tensors.TensorFn], abc.ABC):
    def __init__(self) -> None:
        super().__init__()

        with fake.enable():
            fake_result = self.do()

        assert all(fake.is_fake_tensor(t) for t in fake_result.values())
        self.__attrs = meta.attr_set(fake_result)

    @typing.overload
    def __getitem__(self, key: str) -> tensors.TensorFn: ...

    @typing.overload
    def __getitem__(self, key: list[str]) -> TensorDictFn: ...

    @typing.no_type_check
    def __getitem__(self, key):
        if isinstance(key, str):
            self.__check_keys(key)

            def get_col(tdict: td.TensorDict) -> torch.Tensor:
                return tdict[key]

            return LambdaTensorFn(self, get_col)

        if isinstance(key, slice | np.ndarray | torch.Tensor) or _typing.is_list_of(
            int
        )(key):

            def get_rows(tdict: td.TensorDict) -> td.TensorDict:
                return tdict[key]

            return LambdaTensorDictFn(self, get_rows)

        if isinstance(key, tensors.TensorFn):
            return GatherTensorDictFn(self, deferred_index)

        if _typing.is_list_of(str)(key):
            return self.select(*key)

        raise TypeError(f"Does not handle {key=}, {type(key)=}.")

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
    def deps(self) -> tuple[fn.Fn[typing.Any], ...]:
        raise NotImplementedError

    @property
    def attrs(self):
        return self.__attrs

    def rename(self, **renames: str):
        def rename(tdict: td.TensorDict):
            return td.TensorDict(
                {renames.get(key, key): value for key, value in tdict.items()}
            )

        return LambdaTensorDictFn(self, rename)

    def select(self, *keys: str):
        self.__check_keys(*keys)

        def select(tdict: td.TensorDict) -> td.TensorDict:
            return tdict.select(*keys)

        return LambdaTensorDictFn(self, select)

    def zip(self, other: TensorDictFn, /):
        return MergeTensorDictFn(self, other)

    def __check_keys(self, *keys: str):
        for key in keys:
            if key not in self.keys():
                raise KeyError(key)

    @classmethod
    def from_tensordict(cls, data: td.TensorDict) -> TensorDictFn:
        return TensorDictDataFn(data)


def tdict(item: TensorDictFn | td.TensorDict | cabc.Mapping) -> TensorDictFn:
    if isinstance(item, TensorDictFn):
        return item

    if isinstance(item, td.TensorDict):
        return TensorDictFn.from_tensordict(item)

    if isinstance(item, cabc.Mapping):
        tdict = td.TensorDict.from_any(item)
        return TensorDictFn.from_tensordict(tdict)

    raise TypeError(f"Do not know how to handle {type(item)=}.")


class TensorDictDataFn(TensorDictFn):
    "The `fn.Fn` representing a plain `td.TensorDict`."

    def __init__(self, data: td.TensorDict) -> None:
        self.data = data
        self.data.auto_batch_size_()
        assert self.data.ndim, self.data.shape

        super().__init__()

    @typing.override
    def do(self) -> td.TensorDict:
        if fake.is_enabled():
            return fake.to_fake_tensordict(self.data)

        else:
            return self.data

    @typing.override
    def deps(self):
        return ()


@_common.dcls_no_eq
class LambdaTensorDictFn(TensorDictFn):
    "The `fn.Fn` representing arbitrary computation on `td.TensorDict`."

    source: TensorDictFn
    function: cabc.Callable[[td.TensorDict], td.TensorDict]

    def __post_init__(self) -> None:
        super().__init__()

    @typing.override
    def do(self) -> td.TensorDict:
        source = self.source.do()
        return self.function(source)

    @typing.override
    def deps(self):
        return (self.source,)


@_common.dcls_no_eq
class LambdaTensorFn(tensors.TensorFn):
    "The `fn.Fn` representing arbitrary computation on `td.TensorDict`."

    source: TensorDictFn
    function: cabc.Callable[[td.TensorDict], torch.Tensor]

    def __post_init__(self) -> None:
        super().__init__()

    @typing.override
    def forward(self) -> torch.Tensor:
        source = self.source.do()
        return self.function(source)

    @typing.override
    def deps(self):
        return (self.source,)


@_common.dcls_no_eq
class GatherTensorDictFn(TensorDictFn):

    source: TensorDictFn
    index: tensors.TensorFn

    @typing.override
    def do(self):
        source = self.source.do()
        index = self.index.do()
        return source[index]

    @typing.override
    def deps(self):
        return self.source, self.index


@_common.dcls_no_eq
class MergeTensorDictFn(TensorDictFn):

    left: TensorDictFn
    right: TensorDictFn

    @typing.override
    def do(self):
        left = self.left.do()
        right = self.right.do()
        return td.merge_tensordicts(left, right)

    @typing.override
    def deps(self):
        return self.left, self.right
