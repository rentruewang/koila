# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from abc import ABC
from collections.abc import Callable

import numpy as np
from sympy import Expr
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import Tensor

from aioway._errors import AiowayError

from . import _funcs
from .ops import BatchGen, BatchIter, Op1

__all__ = [
    "MapOpBase",
    "PassOp",
    "FuncFilterOp",
    "ExprFilterOp",
    "ProjectOp",
    "FuncOp",
    "RenameOp",
    "ModuleOp",
]


@dcls.dataclass(frozen=True)
class MapOpBase(Op1, ABC):
    @typing.final
    @typing.override
    def apply(self, stream_iter: BatchIter, /) -> BatchGen:
        for block in stream_iter:
            yield self.map(block)

    @abc.abstractmethod
    def map(self, item: TensorDict, /) -> TensorDict:
        """
        Map individual ``Block`` into something else.
        """

        ...


@dcls.dataclass(frozen=True)
class PassOp(MapOpBase, key="PASS"):
    """
    The ``PASS`` operator does nothing to its inputs.
    """

    map = lambda self, item: item
    "Identity function."


@dcls.dataclass(frozen=True)
class FuncFilterOp(MapOpBase, key="FUNC_FILTER"):
    predicate: Callable[[TensorDict], Tensor]
    """
    The batched prediction of which rows to keep for the inputs.
    """

    @typing.override
    def map(self, item: TensorDict) -> TensorDict:
        "Using the function predicate to filter. Works well with ``numpy``."

        pt = self.predicate(item)

        # Just to be extra fault tolerant.
        pred = pt.numpy()

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


@dcls.dataclass(frozen=True)
class ExprFilterOp(MapOpBase, key="EXPR_FILTER"):
    expr: str | Expr
    """
    The expression of the frame.
    """

    @typing.override
    def map(self, item):
        "Filter with ``expr`` based on ``sympy``."

        return _funcs.filter(item, self.expr)


@dcls.dataclass(frozen=True)
class ProjectOp(MapOpBase, key="PROJECT"):
    """
    Select a subset of the columns.
    """

    subset: list[str] | None = None
    """
    The subset to use. If not give, the ``ProjectStream`` would be a null operation.
    """

    def __post_init__(self) -> None:
        subs = self.subset

        if subs is None:
            return

        if not isinstance(subs, list) and all(isinstance(c, str) for c in subs):
            raise ProjectColumnTypeError("Column must be a list of strings.")

    def __hash__(self):
        return super().__hash__()

    @typing.override
    def map(self, item: TensorDict) -> TensorDict:
        "Perform project. If ``subset`` is ``None``, this is a no-op."
        if self.subset is None:
            return item

        return item.select(*self.subset)


@dcls.dataclass(frozen=True)
class FuncOp(MapOpBase, key="FUNC"):
    """
    ``FuncOp`` is an ``Op`` that performs on the input ``TensorDict``,
    and returns the result as a ``TensorDict``.
    """

    func: Callable[[TensorDict], TensorDict]
    """
    The function to apply to.
    """

    @typing.override
    def map(self, item):
        "Calls ``func`` directly."

        return self.func(item)


@dcls.dataclass(frozen=True)
class RenameOp(MapOpBase, key="RENAME"):
    """
    ``RenameOp`` renames a couple of columns, based on the ``renames`` field dict.
    """

    renames: dict[str, str] = dcls.field(default_factory=dict)
    """
    The mapping dictionary names.
    """

    def __hash__(self):
        return super().__hash__()

    @typing.override
    def map(self, item: TensorDict):
        "Renames the item column with the dictionary."

        return _funcs.rename(item, **self.renames)


@dcls.dataclass(frozen=True)
class ModuleOp(FuncOp, key="MODULE"):
    """
    ``ModuleOpOp`` is an ``Op`` that wraps a ``TensorDictModule``,
    and executes it on the input data.

    It is used to execute the module on the input data,
    and return the result as a ``TensorDict``.
    """

    module: TensorDictModule

    @typing.override
    def map(self, item: TensorDict) -> TensorDict:
        "Calls ``TensorDictModule``."

        return self.module(item)


class FilterBatchSizeError(AiowayError, ValueError): ...


class FitlerPredicateDTypeError(AiowayError, ValueError): ...


class ProjectColumnTypeError(AiowayError, TypeError): ...
