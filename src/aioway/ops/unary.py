# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from sympy import Expr
from tensordict.nn import TensorDictModule

from aioway.blocks import Block
from aioway.errors import AiowayError

from .ops import Op1

__all__ = [
    "PassOp",
    "FuncFilterOp",
    "ExprFilterOp",
    "ProjectOp",
    "MapOp",
    "RenameOp",
    "ModuleOp",
]


@dcls.dataclass(frozen=True)
class PassOp(Op1, key="PASS"):
    """
    The ``PASS`` operator does nothing to its inputs.
    """

    map = lambda self, item: item
    "Identity function."


@dcls.dataclass(frozen=True)
class FuncFilterOp(Op1, key="FUNC_FILTER"):
    predicate: Callable[[Block], NDArray]
    """
    The batched prediction of which rows to keep for the inputs.
    """

    @typing.override
    def map(self, item: Block) -> Block:
        "Using the function predicate to filter. Works well with ``numpy``."

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


@dcls.dataclass(frozen=True)
class ExprFilterOp(Op1, key="EXPR_FILTER"):
    expr: str | Expr
    """
    The expression of the frame.
    """

    @typing.override
    def map(self, item):
        "Filter with ``expr`` based on ``sympy``."

        return item.filter(self.expr)


@dcls.dataclass(frozen=True)
class ProjectOp(Op1, key="PROJECT"):
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

    @typing.override
    def map(self, item: Block) -> Block:
        "Perform project. If ``subset`` is ``None``, this is a no-op."
        if self.subset is None:
            return item

        return item[self.subset]


@dcls.dataclass(frozen=True)
class MapOp(Op1, key="MAP"):
    """
    ``MapOp`` is an ``Op`` that performs on the input ``Block``,
    and returns the result as a ``Block``.

    Note:
        Since ``MapOp`` is simply more powerful,
        as ``torch`` ``Module``s are simply functions,
        with the only constraint being that they need to be ``Module``s.
        Do I remove them completely?
    """

    compute: Callable[[Block], Block]

    @typing.override
    def map(self, item):
        "Calls ``compute`` directly."

        return self.compute(item)


@dcls.dataclass(frozen=True)
class RenameOp(Op1, key="RENAME"):
    """
    Rename a couple of columns.
    """

    renames: dict[str, str] = dcls.field(default_factory=dict)
    """
    The mapping dictionary names.
    """

    @typing.override
    def map(self, item):
        "Renames the item column with the dictionary."

        return item.rename(**self.renames)


@dcls.dataclass(frozen=True)
class ModuleOp(Op1, key="MODULE"):
    """
    ``ModuleOpExec`` is an ``Op`` that wraps a ``TensorDictModule``,
    and executes it on the input data.

    It is used to execute the module on the input data,
    and return the result as a ``Block``.
    """

    module: TensorDictModule

    @typing.override
    def map(self, item):
        "Calls ``TensorDictModule``."

        return self.module(item)


class FilterBatchSizeError(AiowayError, ValueError): ...


class FitlerPredicateDTypeError(AiowayError, ValueError): ...


class ProjectColumnTypeError(AiowayError, TypeError): ...
