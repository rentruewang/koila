# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import typing
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from sympy import Expr

from aioway.attrs import AttrSet
from aioway.blocks import Block
from aioway.errors import AiowayError

from .execs import Exec

__all__ = ["FilterPredExec", "FilterExprExec", "MapExec", "RenameExec", "ProjectExec"]


@typing.final
@dcls.dataclass(frozen=True)
class FilterPredExec(Exec, key="FILTER_PRED"):
    exe: Exec
    """
    The input ``Exec`` of the current ``Exec``.
    """

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

    @property
    @typing.override
    def children(self) -> tuple[Exec]:
        return (self.exe,)


@typing.final
@dcls.dataclass(frozen=True)
class FilterExprExec(Exec, key="FILTER_EXPR"):
    exe: Exec
    """
    The input ``Exec`` of the current ``Exec``.
    """

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

    @property
    @typing.override
    def children(self) -> tuple[Exec]:
        return (self.exe,)


# TODO Improve the initialization of this class.
@typing.final
@dcls.dataclass(frozen=True)
class MapExec(Exec, key="MAP"):
    """
    ``MapExec`` converts the input data stream with a custom function.
    """

    exe: Exec
    """
    The input ``Frame`` to perform computation on.
    """

    compute: Callable[[Block], Block]
    """
    The computation on the input frame.
    """

    output: AttrSet
    """
    Output schema of the ``MapFrame``.
    """

    @typing.override
    def __next__(self) -> Block:
        item = next(self.exe)

        if not isinstance(result := self.compute(item), Block):
            raise MapTypeError(f"Output of {self.compute=} should be `Block`.")

        result.require_attrs(self.output)

        return result

    @property
    @typing.override
    def attrs(self) -> AttrSet:
        return self.output

    @property
    @typing.override
    def children(self) -> tuple[Exec]:
        return (self.exe,)


@typing.final
@dcls.dataclass(frozen=True)
class ProjectExec(Exec, key="PROJECT"):
    """
    Select a subset of the columns.
    """

    exe: Exec
    """
    The input ``Exec`` of the current ``Exec``.
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
    def __next__(self) -> Block:
        item = next(self.exe)
        return item if self.subset is None else item[self.subset]

    @property
    @typing.override
    def attrs(self) -> AttrSet:
        schema = self.exe.attrs

        if self.subset is None:
            return schema

        return schema.project(*self.subset)

    @property
    @typing.override
    def children(self) -> tuple[Exec]:
        return (self.exe,)


@typing.final
@dcls.dataclass(frozen=True, init=False)
class RenameExec(Exec, key="RENAME"):
    """
    Rename a couple of columns.
    """

    exe: Exec
    """
    The input ``Exec`` of the current ``Exec``.
    """

    renames: dict[str, str] = dcls.field(default_factory=dict)
    """
    The mapping dictionary names.
    """

    def __init__(self, __exe: Exec, /, **renames: str) -> None:
        # This constructor is provideds.t. `RenameStream`'s renames can be specified as **kwargs,
        # which means they will be variable names, consistent with what `TensorDict` provides.
        #
        # Even though I'm using a Python version with positional only argument,
        # since `Exec` is common, using `__stream` to avoid name collision (in keys).
        object.__setattr__(self, "exe", __exe)
        object.__setattr__(self, "renames", renames)

    @typing.override
    def __next__(self) -> Block:
        item = next(self.exe)
        return item.rename(**self.renames)

    @property
    @typing.override
    def attrs(self) -> AttrSet:
        return self.exe.attrs.rename(**self.renames)

    @property
    @typing.override
    def children(self) -> tuple[Exec]:
        return (self.exe,)


class FilterBatchSizeError(AiowayError, ValueError): ...


class FitlerPredicateDTypeError(AiowayError, ValueError): ...


class MapTypeError(AiowayError, TypeError): ...


class ProjectColumnTypeError(AiowayError, TypeError): ...
