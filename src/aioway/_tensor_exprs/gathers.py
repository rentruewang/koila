# Copyright (c) AIoWay Authors - All Rights Reserved

"Implements, gather, scatter, `__getitem__`."

import typing

import torch

from aioway import _signs

from . import _common, exprs

__all__ = ["GatherTensorExpr"]


_TENSOR_BINOP = _signs.Signature(torch.Tensor, torch.Tensor, torch.Tensor)


@typing.final
@_common.expr_dcls
class GatherTensorExpr(exprs.TensorExpr):
    """
    The tensor that supports `__getitem__`.
    """

    __match_args__ = "tensor", "index"

    tensor: exprs.TensorExpr

    index: exprs.TensorExpr

    def _compute(self) -> torch.Tensor:
        tensor = self.tensor.compute()
        index = self.index.compute()

        with _common.TRACKER(name="__getitem__", signature=_TENSOR_BINOP):
            return tensor[index]

    def _inputs(self):
        return self.tensor, self.index


@typing.final
@_common.expr_dcls
class StaticArrayGatherTensorExpr(exprs.TensorExpr):
    __match_args__ = "tensor", "index"

    tensor: torch.Tensor

    index: exprs.TensorExpr

    def _compute(self) -> torch.Tensor:
        tensor = self.tensor
        index = self.index.compute()

        with _common.TRACKER(name="gather", signature=_TENSOR_BINOP):
            return tensor[index]

    def _inputs(self):
        return (self.index,)


@typing.final
@_common.expr_dcls
class StaticIndexGatherTensorExpr(exprs.TensorExpr):
    __match_args__ = "tensor", "index"

    tensor: exprs.TensorExpr

    index: int | slice | torch.Tensor

    def _compute(self) -> torch.Tensor:
        tensor = self.tensor.compute()
        index = self.index

        with _common.TRACKER(name="__getitem__", signature=_TENSOR_BINOP):
            return tensor[index]

    def _inputs(self):
        return (self.tensor,)
