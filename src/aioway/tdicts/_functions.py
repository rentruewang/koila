# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections import abc as cabc

import tensordict as td
import torch

from aioway import _common, fake, tensors

from . import tdicts

__all__ = ["TensorDictDataFn", "LambdaTensorDictFn", "LambdaTensorFn"]


class TensorDictDataFn(tdicts.TensorDictFn):
    "The `fn.Fn` representing a plain `td.TensorDict`."

    def __init__(self, data: td.TensorDict) -> None:
        self.data = data
        self.data.auto_batch_size_()

        super().__init__()

    @typing.override
    def do(self) -> td.TensorDict:
        if not (mode := fake.is_enabled()):
            return self.data

        converter = mode.fake_tensor_converter
        return td.TensorDict(
            {
                key: converter.from_real_tensor(mode, value)
                for key, value in self.data.items()
            }
        )

    @typing.override
    def deps(self):
        return ()


@_common.dcls_no_eq
class LambdaTensorDictFn(tdicts.TensorDictFn):
    "The `fn.Fn` representing arbitrary computation on `td.TensorDict`."

    source: tdicts.TensorDictFn
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

    source: tdicts.TensorDictFn
    function: cabc.Callable[[td.TensorDict], torch.Tensor]

    def __post_init__(self) -> None:
        super().__init__()

    @typing.override
    def do(self) -> torch.Tensor:
        source = self.source.do()
        return self.function(source)

    @typing.override
    def deps(self):
        return (self.source,)
