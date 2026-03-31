# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import tensordict as td
import torch

from aioway import _common, tensors

from . import tdicts

__all__ = ["GetItemFn", "SelectFn"]


@_common.dcls_no_eq
class GetItemFn(tensors.TensorFn):
    tdict: tdicts.TensorDictFn
    column: str

    def __post_init__(self):
        super().__init__()

    @typing.override
    def forward(self) -> torch.Tensor:
        tdict = self.tdict.do()
        return tdict[self.column]

    @typing.override
    def _deps(self):
        yield self.tdict


@_common.dcls_no_eq
class SelectFn(tdicts.TensorDictFn):
    tdict: tdicts.TensorDictFn
    columns: list[str]

    def __post_init__(self):
        super().__init__()

    @typing.override
    def forward(self) -> td.TensorDict:
        tdict = self.tdict.do()
        return tdict.select(*self.columns)

    @typing.override
    def _deps(self):
        yield self.tdict
