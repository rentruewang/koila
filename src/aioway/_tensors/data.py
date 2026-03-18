# Copyright (c) AIoWay Authors - All Rights Reserved

"The 0-ary, 1-ary expressions."

import typing
from collections.abc import KeysView
from typing import ClassVar

from numpy.typing import NDArray
from tensordict import TensorDict
from torch import Tensor

from aioway import _logging
from aioway._exprs import Expr

from . import _common
from .exprs import TensorDictExpr, TensorExpr

__all__ = [
    "SourceTensorDictExpr",
    "SourceTensorExpr",
    "CacheTensorDictExpr",
    "CacheTensorExpr",
    "ColumnTensorExpr",
    "BatchTensorDictExpr",
    "ItemTensorDictExpr",
]

LOGGER = _logging.get_logger(__name__)


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
class _CacheExpr[T: Tensor | TensorDict]:
    __match_args__ = ()

    data: T

    def __init__(self, expr: Expr[T]) -> None:
        self.data = expr.compute()

    def _compute(self) -> T:
        return self.data

    def _inputs(self):
        return ()


@_common.expr_dcls
class CacheTensorDictExpr(_CacheExpr[TensorDict], TensorDictExpr): ...


@_common.expr_dcls
class CacheTensorExpr(_CacheExpr[Tensor], TensorExpr): ...


@_common.expr_dcls
class ColumnTensorExpr(TensorExpr):
    source: TensorDictExpr

    column: str

    @typing.no_type_check
    def _compute(self) -> Tensor:
        return self.source.compute()[self.column]

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
        return self.source.compute()[self.index]

    def _inputs(self) -> tuple[Expr[TensorDict], ...]:
        return (self.source,)


@_common.expr_dcls
class ItemTensorDictExpr(_GetItemTensorExpr[int], TensorDictExpr): ...


type BatchIndex = list[int] | slice | NDArray | Tensor


@_common.expr_dcls
class BatchTensorDictExpr(_GetItemTensorExpr[BatchIndex], TensorDictExpr):
    def keys(self) -> KeysView[str]:
        return self.source.keys()
