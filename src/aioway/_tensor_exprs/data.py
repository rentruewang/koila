# Copyright (c) AIoWay Authors - All Rights Reserved

"The 0-ary, 1-ary expressions."

import typing
from collections import abc as cabc

import tensordict as td
import torch
from numpy import typing as npt

from aioway import _signs
from aioway._tracking import logging

from . import _common, exprs

__all__ = [
    "SourceTensorDictExpr",
    "SourceTensorExpr",
    "ColumnTensorExpr",
    "BatchTensorDictExpr",
    "ItemTensorDictExpr",
]

LOGGER = logging.get_logger(__name__)


@_common.expr_dcls
class _SourceExpr[T: torch.Tensor | td.TensorDict]:
    _DATA_TYPE: typing.ClassVar[type[T]]

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
            name="source", signature=_signs.Signature(self._DATA_TYPE, self._DATA_TYPE)
        ):
            return self.data

    def _inputs(self):
        return ()


@_common.expr_dcls
class SourceTensorDictExpr(_SourceExpr[td.TensorDict], exprs.TensorDictExpr):
    "The source expression for `td.TensorDict`."

    _DATA_TYPE = td.TensorDict

    def keys(self) -> cabc.KeysView[str]:
        return self.data.keys()


@_common.expr_dcls
class SourceTensorExpr(_SourceExpr[torch.Tensor], exprs.TensorExpr):
    "The source expression for `torch.Tensor`."

    _DATA_TYPE = torch.Tensor


@_common.expr_dcls
class ColumnTensorExpr(exprs.TensorExpr):
    source: exprs.TensorDictExpr

    column: str

    @typing.no_type_check
    def _compute(self) -> torch.Tensor:
        pulled = self.source.compute()

        with _common.TRACKER(
            name="column", signature=_signs.Signature(td.TensorDict, torch.Tensor)
        ):
            return pulled[self.column]

    def _inputs(self):
        return (self.source,)


@_common.expr_dcls
class _GetItemTensorExpr[T](exprs.TensorDictExpr):
    source: exprs.TensorDictExpr

    index: T

    def keys(self) -> cabc.KeysView[str]:
        return self.source.keys()

    @typing.no_type_check
    def _compute(self) -> td.TensorDict:
        pulled = self.source.compute()

        with _common.TRACKER(
            name="__getitem__",
            signature=_signs.Signature(td.TensorDict, type(self.index)),
        ):
            return pulled[self.index]

    def _inputs(self):
        return (self.source,)


@_common.expr_dcls
class ItemTensorDictExpr(_GetItemTensorExpr[int], exprs.TensorDictExpr): ...


type BatchIndex = list[int] | slice | npt.NDArray | torch.Tensor


@_common.expr_dcls
class BatchTensorDictExpr(_GetItemTensorExpr[BatchIndex], exprs.TensorDictExpr):
    def keys(self) -> cabc.KeysView[str]:
        return self.source.keys()
