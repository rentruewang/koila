# Copyright (c) AIoWay Authors - All Rights Reserved

"The 0-ary, 1-ary expressions."

import typing
from collections.abc import KeysView
from typing import ClassVar

from numpy.typing import NDArray
from tensordict import TensorDict
from torch import Tensor

from aioway._signs import Signature
from aioway._tracking import logging

from . import _common
from .exprs import TensorDictExpr, TensorExpr

__all__ = [
    "SourceTensorDictExpr",
    "SourceTensorExpr",
    "ColumnTensorExpr",
    "BatchTensorDictExpr",
    "ItemTensorDictExpr",
]

LOGGER = logging.get_logger(__name__)


@_common.expr_dcls
class _SourceExpr[T: Tensor | TensorDict]:
    _DATA_TYPE: ClassVar[type[T]]

    __match_args__ = ()

    data: T

    def __post_init__(self):
        if not isinstance(self.data, self._DATA_TYPE):
            raise TypeError(f"Expected type {self._DATA_TYPE}, got {type(self.data)}.")

    def __repr__(self):
        shape = self.data.shape
        dtype = self.data.dtype
        device = self.data.device
        return f"{self.__class__.__name__}({shape=}, {dtype=}, {device=})"

    def _compute(self) -> T:
        with _common.TRACKER(
            name="source", signature=Signature(self._DATA_TYPE, self._DATA_TYPE)
        ):
            return self.data

    def _inputs(self):
        return ()


@_common.expr_dcls
class SourceTensorDictExpr(_SourceExpr[TensorDict], TensorDictExpr):
    "The source expression for `TensorDict`."

    _DATA_TYPE = TensorDict

    def keys(self) -> KeysView[str]:
        return self.data.keys()


@_common.expr_dcls
class SourceTensorExpr(_SourceExpr[Tensor], TensorExpr):
    "The source expression for `Tensor`."

    _DATA_TYPE = Tensor


@_common.expr_dcls
class ColumnTensorExpr(TensorExpr):
    source: TensorDictExpr

    column: str

    @typing.no_type_check
    def _compute(self) -> Tensor:
        pulled = self.source.compute()

        with _common.TRACKER(name="column", signature=Signature(TensorDict, Tensor)):
            return pulled[self.column]

    def _inputs(self):
        return (self.source,)


@_common.expr_dcls
class _GetItemTensorExpr[T](TensorDictExpr):
    source: TensorDictExpr

    index: T

    def keys(self) -> KeysView[str]:
        return self.source.keys()

    @typing.no_type_check
    def _compute(self) -> TensorDict:
        pulled = self.source.compute()

        with _common.TRACKER(
            name="__getitem__", signature=Signature(TensorDict, type(self.index))
        ):
            return pulled[self.index]

    def _inputs(self):
        return (self.source,)


@_common.expr_dcls
class ItemTensorDictExpr(_GetItemTensorExpr[int], TensorDictExpr): ...


type BatchIndex = list[int] | slice | NDArray | Tensor


@_common.expr_dcls
class BatchTensorDictExpr(_GetItemTensorExpr[BatchIndex], TensorDictExpr):
    def keys(self) -> KeysView[str]:
        return self.source.keys()
