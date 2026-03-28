# Copyright (c) AIoWay Authors - All Rights Reserved

"Implements, gather, scatter, `__getitem__`."

import typing

from torch import Tensor

from aioway._signs import Signature

from . import _common
from .exprs import TensorExpr

__all__ = ["GatherTensorExpr"]


_TENSOR_BINOP = Signature(Tensor, Tensor, Tensor)


@typing.final
@_common.expr_dcls
class GatherTensorExpr(TensorExpr):
    """
    The tensor that supports `__getitem__`.
    """

    __match_args__ = "tensor", "index"

    tensor: TensorExpr

    index: TensorExpr

    def _compute(self) -> Tensor:
        tensor = self.tensor.compute()
        index = self.index.compute()

        with _common.TRACKER(name="__getitem__", signature=_TENSOR_BINOP):
            return tensor[index]

    def _inputs(self):
        return self.tensor, self.index


@typing.final
@_common.expr_dcls
class StaticArrayGatherTensorExpr(TensorExpr):
    __match_args__ = "tensor", "index"

    tensor: Tensor

    index: TensorExpr

    def _compute(self) -> Tensor:
        tensor = self.tensor
        index = self.index.compute()

        with _common.TRACKER(name="gather", signature=_TENSOR_BINOP):
            return tensor[index]

    def _inputs(self):
        return (self.index,)


@typing.final
@_common.expr_dcls
class StaticIndexGatherTensorExpr(TensorExpr):
    __match_args__ = "tensor", "index"

    tensor: TensorExpr

    index: int | slice | Tensor

    def _compute(self) -> Tensor:
        tensor = self.tensor.compute()
        index = self.index

        with _common.TRACKER(name="__getitem__", signature=_TENSOR_BINOP):
            return tensor[index]

    def _inputs(self):
        return (self.tensor,)
