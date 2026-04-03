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
class LambdaTensorFn(tensors.BasicPreviewFn):
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


@_common.dcls_no_eq
class GatherTensorDictFn(tdicts.TensorDictFn):

    source: tdicts.TensorDictFn
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
class MergeTensorDictFn(tdicts.TensorDictFn):

    left: tdicts.TensorDictFn
    right: tdicts.TensorDictFn

    @typing.override
    def do(self):
        left = self.left.do()
        right = self.right.do()
        return td.merge_tensordicts(left, right)

    @typing.override
    def deps(self):
        return self.left, self.right
