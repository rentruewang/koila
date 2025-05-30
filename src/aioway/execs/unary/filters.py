# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from sympy import Expr

from aioway.attrs import AttrSet
from aioway.blocks import Block
from aioway.errors import AiowayError

from .unary import UnaryExec

__all__ = ["FilterPredExec", "FilterExprExec"]


@typing.final
@dcls.dataclass
class FilterPredExec(UnaryExec, key="FILTER_PRED"):
    predicate: Callable[[Block], NDArray]
    """
    The batched prediction of which rows to keep for the inputs.
    """

    @typing.override
    def __next__(self) -> Block:
        item = next(self.exe)
        pred = self.predicate(item)

        # Just to be extra fault tolerant.
        pred = np.array(pred)

        # Convert to int indices.
        if np.isdtype(pred.dtype, "bool"):
            if len(item) != len(pred):
                raise FilterBatchSizeError(
                    f"The output length of {self.predicate=} does not match the input, "
                    "even though a boolean array is returned."
                )
            pred = np.arange(len(item))[pred]

        # Must be integer array for indices.
        if not np.isdtype(pred.dtype, "integral"):
            raise FitlerPredicateDTypeError(
                f"Output of {self.predicate=} should be an integer array."
            )

        return item[pred]

    @property
    @typing.override
    def attrs(self) -> AttrSet:
        return self.exe.attrs


@typing.final
@dcls.dataclass
class FilterExprExec(UnaryExec, key="FILTER_EXPR"):
    expr: str | Expr
    """
    The expression of the frame.
    """

    @typing.override
    def __next__(self) -> Block:
        item = next(self.exe)
        return item.filter(self.expr)

    @property
    @typing.override
    def attrs(self) -> AttrSet:
        return self.exe.attrs


class FilterBatchSizeError(AiowayError, ValueError): ...


class FitlerPredicateDTypeError(AiowayError, ValueError): ...
