# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import operator
import typing
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from sympy import Expr

from aioway.blocks import Block
from aioway.datatypes import AttrSet
from aioway.errors import AiowayError

from .streams import Stream

__all__ = [
    "FilterPredStream",
    "FilterExprStream",
    "MapStream",
    "RenameStream",
    "ProjectStream",
]


@typing.final
@dcls.dataclass(frozen=True)
class FilterPredStream(Stream):
    stream: Stream
    """
    The input ``Stream`` of the current ``Stream``.
    """

    predicate: Callable[[Block], NDArray]
    """
    The batched prediction of which rows to keep for the inputs.
    """

    @typing.override
    def __next__(self) -> Block:
        item = next(self.stream)
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

    @typing.override
    def __length_hint__(self):
        return operator.length_hint(self.stream)

    @property
    def attrs(self) -> AttrSet:
        return self.stream.attrs


@typing.final
@dcls.dataclass(frozen=True)
class FilterExprStream(Stream):
    stream: Stream
    """
    The input ``Stream`` of the current ``Stream``.
    """

    expr: str | Expr
    """
    The expression of the frame.
    """

    @typing.override
    def __next__(self) -> Block:
        item = next(self.stream)
        return item.filter(self.expr)

    @typing.override
    def __length_hint__(self):
        return operator.length_hint(self.stream)

    @property
    def attrs(self) -> AttrSet:
        return self.stream.attrs


@typing.final
@dcls.dataclass(frozen=True)
class MapStream(Stream):
    """
    ``MapStream`` converts the input data stream with a custom function.

    Todo:
        Improve the initialization of this class.
    """

    stream: Stream
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
        item = next(self.stream)

        if not isinstance(result := self.compute(item), Block):
            raise MapTypeError(f"Output of {self.compute=} should be `Block`.")

        result.must_have_attrs(self.output)

        return result

    @typing.override
    def __length_hint__(self):
        return operator.length_hint(self.stream)

    @property
    def attrs(self) -> AttrSet:
        return self.output


@typing.final
@dcls.dataclass(frozen=True)
class ProjectStream(Stream):
    """
    Select a subset of the columns.
    """

    stream: Stream
    """
    The input ``Stream`` of the current ``Stream``.
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
        item = next(self.stream)
        return item if self.subset is None else item[self.subset]

    @property
    def attrs(self) -> AttrSet:
        schema = self.stream.attrs

        if self.subset is None:
            return schema

        return schema.project(*self.subset)


@typing.final
@dcls.dataclass(frozen=True, init=False)
class RenameStream(Stream):
    """
    Rename a couple of columns.
    """

    stream: Stream
    """
    The input ``Stream`` of the current ``Stream``.
    """

    renames: dict[str, str] = dcls.field(default_factory=dict)
    """
    The mapping dictionary names.
    """

    def __init__(self, __stream: Stream, /, **renames: str) -> None:
        # This constructor is provideds.t. ``RenameStream``'s renames can be specified as **kwargs,
        # which means they will be variable names, consistent with what ``TensorDict`` provides.
        #
        # Even though I'm using a Python version with positional only argument,
        # since ``stream`` is common, using ``__stream`` to avoid name collision (in keys).
        object.__setattr__(self, "stream", __stream)
        object.__setattr__(self, "renames", renames)

    @typing.override
    def __next__(self) -> Block:
        item = next(self.stream)
        return item.rename(**self.renames)

    @typing.override
    def __length_hint__(self):
        return operator.length_hint(self.stream)

    @property
    def attrs(self) -> AttrSet:
        return self.stream.attrs.rename(**self.renames)


class FilterBatchSizeError(AiowayError, ValueError): ...


class FitlerPredicateDTypeError(AiowayError, ValueError): ...


class MapTypeError(AiowayError, TypeError): ...


class ProjectColumnTypeError(AiowayError, TypeError): ...
