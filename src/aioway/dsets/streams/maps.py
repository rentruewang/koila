# Copyright (c) AIoWay Authors - All Rights Reserved

"The ``Stream``s that apply a transformation on the input ``Stream``."

import abc
import dataclasses as dcls
import typing
from abc import ABC
from collections.abc import Callable

import torch
from sympy import Expr
from tensordict import TensorDict
from torch import Tensor

from . import _funcs
from .streams import Stream

__all__ = [
    "MapStream",
    "ApplyStream",
    "FuncFilterStream",
    "ExprFilterStream",
    "ProjectStream",
    "RenameStream",
]


@dcls.dataclass
class MapStream(Stream, ABC):
    """
    The shared base class for all the ``map`` like ``Stream``s,
    which share the trait of::

        #. Having 1 child, named ``source``.
        #. Calls ``next`` on its ``source`` once per ``next``.
        #. Can be represented as a pure, 1 argument function.
        #. Input to output is a batch-to-batch function.

    These traits are shared in this base class.

    .. note::
        Though having a 1-1 input to output batch count, this is considered to be a ``flat_map``,
        where each input row can correspond to one or multiple or 0 rows, in the same minibatch.
    """

    source: Stream
    """
    The source stream that will be yielded from.
    """

    def __post_init__(self):
        if not isinstance(self.source, Stream):
            raise ValueError(
                f"{self.source=} should have been a `Stream`. Got {type(self.source)=}"
            )

    @property
    @typing.override
    def size(self) -> int:
        "This stream should have about the same length as the input."

        return self.source.size

    @abc.abstractmethod
    def _apply(self, batch: TensorDict) -> TensorDict:
        """
        The protected method that subclass should overwrite.
        This method will define how each batch is processed.

        Args:
            batch: The batch to handle. Will be a ``TensorDict``.

        Returns:
            Another ``TensorDict``. Does not need to have the same ``__len__`` to the input.
            See class docstring for more details.
        """

        ...

    @typing.override
    @typing.final
    def _read(self) -> TensorDict:
        # A ``map`` kind of ``Stream`` always calls ``next`` once on its source.
        # May raise ``StopIteration`` here.
        next_batch = next(self.source)
        return self._apply(next_batch)

    @typing.final
    @typing.override
    def _children(self):
        yield self.source


@dcls.dataclass
class ApplyStream(MapStream):
    """
    A ``Stream`` that you can customize what the ``__next__`` function do.

    The full loop would be something like:

    .. code-block:: python

        for batch in self.source:
            yield self.apply(batch)
    """

    apply: Callable[[TensorDict], TensorDict]
    """
    Compute the output of ``__next__`` based on the input.
    """

    @typing.override
    def _apply(self, batch: TensorDict) -> TensorDict:
        return self.apply(batch)


@dcls.dataclass
class FuncFilterStream(MapStream):
    """
    A ``Stream`` that filteres on its inputs, based on a preducate function.

    The input is being used to generate predicate,
    and the output of predicate must be a boolean ``Tensor`` of the same length as the input.

    .. code-block:: python

        for batch in self.source:
            yield batch[self.predicate(batch)]
    """

    predicate: Callable[[TensorDict], Tensor]
    """
    A function of ``TensorDict -> Tensor``.
    """

    @typing.override
    def _apply(self, batch: TensorDict) -> TensorDict:
        pred = self.predicate(batch)

        if pred.dtype is not torch.bool:
            raise ValueError()

        return batch[pred]


@dcls.dataclass
class ExprFilterStream(MapStream):
    """
    A ``Stream`` that filteres on its inputs, based on a preducate expression.

    The expression represents the columns in the ``TensorDict``.

    For example, an expression ``a == 0`` can be thought of as ``batch["a"] == 0``,
    which returns a boolean ``Tensor``, and can be used to index the ``TensorDict``.
    """

    predicate: str | Expr
    """
    A string of a ``sympy.Expr``.
    """

    @typing.override
    def _apply(self, batch: TensorDict) -> TensorDict:
        return _funcs.filter(batch, self.predicate)


@dcls.dataclass
class ProjectStream(MapStream):
    """
    Projection of the input table. The ``subset`` should be a subset of the input columns.
    """

    subset: list[str] = dcls.field(default_factory=list)
    """
    Input columns that appears in the outputs.
    """

    @typing.override
    def _apply(self, batch: TensorDict) -> TensorDict:
        if not self.subset:
            return batch

        return batch.select(*self.subset)


@dcls.dataclass
class RenameStream(MapStream):
    """
    Renames some columns in the inputs in the outputs.
    """

    renames: dict[str, str] = dcls.field(default_factory=dict)
    """
    Columns to rename. Mapping from original to the new names.
    """

    @typing.override
    def _apply(self, batch: TensorDict) -> TensorDict:
        if not self.renames:
            return batch

        return _funcs.rename(batch, **self.renames)
