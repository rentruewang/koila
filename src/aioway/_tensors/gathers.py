# Copyright (c) AIoWay Authors - All Rights Reserved

"Implements, gather, scatter, `__getitem__`."

import typing

from torch import Tensor

from aioway._typing import BatchIndex

from . import _common
from .exprs import TensorExpr

__all__ = ["GatherTensorExpr"]


@typing.final
@_common.expr_dcls
class GatherTensorExpr(TensorExpr):
    """
    The tensor that supports `__getitem__`.
    """

    __match_args__ = "array", "index"

    array: TensorExpr

    index: TensorExpr

    def _compute(self) -> Tensor:
        tensor = self.array.compute()
        index = self.index.compute()

        return tensor[index]

    def _inputs(self):
        return self.array, self.index


@typing.final
@_common.expr_dcls
class StaticArrayGatherTensorExpr:
    __match_args__ = "array", "index"

    array: Tensor

    index: TensorExpr

    def _compute(self) -> Tensor:
        tensor = self.array
        index = self.index.compute()

        return tensor[index]

    def _inputs(self):
        return (self.index,)


@typing.final
@_common.expr_dcls
class StaticIndexGatherTensorExpr:
    __match_args__ = "array", "index"

    array: TensorExpr

    index: int | BatchIndex

    def _compute(self) -> Tensor:
        tensor = self.array.compute()
        index = self.index

        return tensor[index]

    def _inputs(self):
        return (self.array,)
