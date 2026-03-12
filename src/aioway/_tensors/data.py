# Copyright (c) AIoWay Authors - All Rights Reserved

"The 0-ary expressions, and projection."

import dataclasses as dcls
from collections.abc import KeysView, Sequence

from tensordict import TensorDict
from torch import Tensor

from aioway._exprs import Expr

from .exprs import TensorDictExpr, TensorExpr

__all__ = [
    "SourceTensorDictExpr",
    "SourceTensorExpr",
    "CacheTensorDictExpr",
    "CacheTensorExpr",
    "GetItemTensorExpr",
    "SelectTensorDictExpr",
]


@dcls.dataclass(frozen=True, match_args=False)
class _SourceExpr[T: Tensor | TensorDict]:
    __match_args__ = ()

    data: T

    def _compute(self) -> T:
        return self.data

    def _inputs(self):
        return ()


@dcls.dataclass(frozen=True, match_args=False)
class SourceTensorDictExpr(_SourceExpr[TensorDict], TensorDictExpr):
    "The source expression for ``TensorDict``."

    ...


@dcls.dataclass(frozen=True, match_args=False)
class SourceTensorExpr(_SourceExpr[Tensor], TensorExpr):
    "The source expression for ``Tensor``."

    ...


@dcls.dataclass(init=False, match_args=False)
class _CacheExpr[T: Tensor | TensorDict]:
    __match_args__ = ()

    data: T

    def __init__(self, expr: Expr[T]) -> None:
        self.data = expr.compute()

    def _compute(self) -> T:
        return self.data

    def _inputs(self):
        return ()


@dcls.dataclass(init=False, match_args=False)
class CacheTensorDictExpr(_CacheExpr[TensorDict], TensorDictExpr): ...


@dcls.dataclass(init=False, match_args=False)
class CacheTensorExpr(_CacheExpr[Tensor], TensorExpr): ...


@dcls.dataclass(frozen=True)
class GetItemTensorExpr(TensorExpr):
    source: TensorDictExpr

    column: str

    def _compute(self) -> Tensor:
        return self.source.compute()[self.column]

    def _inputs(self):
        return (self.source,)


@dcls.dataclass(frozen=True)
class SelectTensorDictExpr(TensorDictExpr):
    source: TensorDictExpr

    columns: Sequence[str]

    def keys(self) -> KeysView[str]:
        raise NotImplementedError

    def _compute(self) -> TensorDict:
        return self.source.compute().select(*self.columns)

    def _inputs(self):
        return (self.source,)
