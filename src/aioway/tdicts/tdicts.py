# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing
from collections import abc as cabc

import numpy as np
import tensordict as td
import torch

from aioway import _typing, fake, fn, tensors

from . import attrs

__all__ = ["TensorDictFn", "tdict"]


class TensorDictFn(fn.Fn[td.TensorDict], cabc.Mapping[str, tensors.TensorFn], abc.ABC):
    def __init__(self) -> None:
        super().__init__()

        with fake.enable():
            fake_result = self.do()

        assert all(fake.is_fake_tensor(t) for t in fake_result.values())
        self.__attrs = attrs.attr_set(fake_result)

    @typing.overload
    def __getitem__(self, key: str) -> tensors.TensorFn: ...

    @typing.overload
    def __getitem__(self, key: list[str]) -> TensorDictFn: ...

    @typing.no_type_check
    def __getitem__(self, key):
        from . import _functions

        if isinstance(key, str):

            def get_col(tdict: td.TensorDict) -> torch.Tensor:
                return tdict[key]

            return _functions.LambdaTensorFn(self, get_col)

        if isinstance(key, slice | np.ndarray | torch.Tensor) or _typing.is_list_of(
            int
        )(key):

            def get_rows(tdict: td.TensorDict) -> td.TensorDict:
                return tdict[key]

            return _functions.LambdaTensorDictFn(self, get_rows)

        if isinstance(key, tensors.TensorFn):
            return _functions.GatherTensorDictFn(self, deferred_index)

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
        from . import _functions

        def rename(tdict: td.TensorDict):
            return td.TensorDict(
                {renames.get(key, key): value for key, value in tdict.items()}
            )

        return _functions.LambdaTensorDictFn(self, rename)

    def select(self, *keys: str):
        from . import _functions

        def select(tdict: td.TensorDict) -> td.TensorDict:
            return tdict.select(*keys)

        return _functions.LambdaTensorDictFn(self, select)

    def zip(self, other: TensorDictFn, /):
        from . import _functions

        return _functions.MergeTensorDictFn(self, other)

    @classmethod
    def from_tensordict(cls, data: td.TensorDict) -> TensorDictFn:
        from . import _functions

        return _functions.TensorDictDataFn(data)


def tdict(item: TensorDictFn | td.TensorDict | cabc.Mapping) -> TensorDictFn:
    if isinstance(item, TensorDictFn):
        return item

    if isinstance(item, td.TensorDict):
        return TensorDictFn.from_tensordict(item)

    if isinstance(item, cabc.Mapping):
        tdict = td.TensorDict.from_any(item)
        return TensorDictFn.from_tensordict(tdict)

    raise TypeError(f"Do not know how to handle {type(item)=}.")
