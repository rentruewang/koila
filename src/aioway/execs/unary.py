# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from abc import ABC
from collections.abc import Callable, Iterator
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from sympy import Expr
from tensordict.nn import TensorDictModule

from aioway.blocks import Block
from aioway.errors import AiowayError

from .execs import Exec

__all__ = [
    "Exec1",
    "PassExec",
    "FuncFilterExec",
    "ExprFilterExec",
    "ProjectExec",
    "MapExec",
    "RenameExec",
    "ModuleExec",
]


@dcls.dataclass(frozen=True)
class Exec1(Exec, ABC):
    ARGC: ClassVar[int] = 1

    child: Exec
    """
    The child executor.
    """

    @typing.override
    def __iter__(self) -> Iterator[Block]:
        for item in self.child:
            yield self.map(item)

    @abc.abstractmethod
    def map(self, item: Block, /) -> Block: ...

    @property
    @typing.final
    def children(self) -> tuple[Exec]:
        return (self.child,)


@dcls.dataclass(frozen=True)
class PassExec(Exec1, key="PASS_1"):
    """
    The ``PASS`` operator does nothing to its inputs.
    """

    @typing.override
    def map(self, item: Block) -> Block:
        return item


@dcls.dataclass(frozen=True)
class FuncFilterExec(Exec1, key="FUNC_FILTER_1"):
    predicate: Callable[[Block], NDArray]
    """
    The batched prediction of which rows to keep for the inputs.
    """

    @typing.override
    def map(self, item: Block) -> Block:
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
class ExprFilterExec(Exec1, key="EXPR_FILTER_1"):
    expr: str | Expr
    """
    The expression of the frame.
    """

    @typing.override
    def map(self, item: Block) -> Block:
        return item.filter(self.expr)


@dcls.dataclass(frozen=True)
class ProjectExec(Exec1, key="PROJECT_1"):
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
        if self.subset is None:
            return item

        return item[self.subset]


@dcls.dataclass(frozen=True)
class MapExec(Exec1, key="MAP_1"):
    """
    ``MapOp`` is an ``Op`` that performs on the input ``Block``,
    and returns the result as a ``Block``.

    Note:
        Since ``MapOp`` is simply more powerful,
        as ``torch`` ``Module``s are simply functions,
        with the only constraint being that they need to be ``Module``s.
        Do I remove this completely?
    """

    compute: Callable[[Block], Block]

    def map(self, item: Block) -> Block:
        return self.compute(item)


@dcls.dataclass(frozen=True)
class RenameExec(Exec1, key="RENAME_1"):
    """
    Rename a couple of columns.
    """

    renames: dict[str, str] = dcls.field(default_factory=dict)
    """
    The mapping dictionary names.
    """

    @typing.override
    def map(self, item: Block) -> Block:
        return item.rename(**self.renames)


@dcls.dataclass(frozen=True)
class ModuleExec(Exec1, key="MODULE_1"):
    """
    ``ModuleOpExec`` is an ``Op`` that wraps a ``TensorDictModule``,
    and executes it on the input data.

    It is used to execute the module on the input data,
    and return the result as a ``Block``.
    """

    module: TensorDictModule

    def map(self, item: Block) -> Block:
        self.module(item)
        return item


class FilterBatchSizeError(AiowayError, ValueError): ...


class FitlerPredicateDTypeError(AiowayError, ValueError): ...


class ProjectColumnTypeError(AiowayError, TypeError): ...
