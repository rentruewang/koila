# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from torch import Tensor

from aioway import _common
from aioway.tensors import TensorFn

from .fn import TensorDictFn

__all__ = ["GetItemFn"]


@_common.dcls_no_eq
class GetItemFn(TensorFn):
    tdict: TensorDictFn
    column: str

    def __post_init__(self):
        super().__init__()

    @typing.override
    def forward(self) -> Tensor:
        tdict = self.tdict.do()
        return tdict[self.column]

    @typing.override
    def _deps(self):
        yield self.tdict


@_common.dcls_no_eq
class SelectFn(TensorFn):
    tdict: TensorDictFn
    columns: list[str]

    def __post_init__(self):
        super().__init__()

    @typing.override
    def forward(self) -> Tensor:
        tdict = self.tdict.do()
        return tdict.select(*self.columns)

    @typing.override
    def _deps(self):
        yield self.tdict
