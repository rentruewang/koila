# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import json
import logging
import typing
from abc import ABC
from collections.abc import Generator, Iterable, Iterator
from typing import ClassVar, Protocol

from tensordict import TensorDict

from .. import thunks
from ..thunks import Thunk

__all__ = ["Plan", "Plan0", "Plan1", "Plan2", "BatchIter", "BatchGen"]

LOGGER = logging.getLogger(__name__)

type BatchIter = Iterable[TensorDict]
"""
The function ``__iter__`` of ``TensorDict``.
Would create a new ``Generator`` everytime ``iter`` is called.
"""

type BatchGen = Generator[TensorDict]
"""
``Generator`` of ``TensorDict``.
"""


@dcls.dataclass(frozen=True)
class Plan(ABC):
    """
    ``Plan`` is the **pure**, **atomic** operation, be it operator or operand, that works on data.

    This means it has the following semantics::

        1. It doesn't store state.
        The same ``Plan`` can be applied multiple times, and used in comparison.
        2. Running on 1 device, and does not cross device boundary.
        3. Isolated in the computation graph, and can be rearranged.


    An operation is essentially an generator function over data (``Block``),
    but itself doesn't store which previous node it would be operating on.
    This design allows users to write ``Plan`` imperatively as generators,
    rather than iterators (``__next__`` function) with state management.

    There are 3 kinds of operators, ``Plan0``, ``Plan1``, ``Plan2``,
    representing 0-ary, 1-ary, 2-ary operators respectively.
    """

    ARGC: ClassVar[int]
    """
    Argument count of the current node.
    """

    class Visitor[T](Protocol):
        """
        The ``Visitor`` pattern / strategy for ``Plan``.
        """

        def plan_0(self, plan: "Plan0", /) -> T: ...

        def plan_1(self, plan: "Plan1", /) -> T: ...

        def plan_2(self, plan: "Plan2", /) -> T: ...

    def __hash__(self) -> int:
        """
        This is s.t. we can use ``Plan`` in dictionary lookup.
        """

        return hash(json.dumps(dcls.asdict(self)))

    @typing.no_type_check
    @abc.abstractmethod
    def apply(self, *ops: BatchIter) -> BatchGen:
        """
        The ``apply`` method launches a new ``Generator`` to loop over the inputs.
        Every call is creates / rebuilds brand new computation.

        Using ``Generator`` to allow possible future 2 way communication into coroutine.

        We want the subclasses signatures to be::

            #. For ``op0: Plan0``, ``op0()``.
            #. For ``op1: Plan1``, ``op1(child)``.
            #. For ``op2: Plan2``, ``op2(left, right)``.

        Args:
            *ops: The operands to take (would be unpacked in subclass).

        Returns:
            A stream of ``Block``s.

        Note:
            Why is it that this is an ``__iter__`` method,
            rather than a ``__next__`` method like a lot of other RDBMS implementations?

            The reasons are as follows:

            0. This is not RDBMS, just takes inspiration.
                Let's have some style, right?

            1. Imperative is better looking.
                Easier to read rather than having to resort to state management.

            2. Combined iteration and evaluation strategy.
                This can be useful in not writing the functions twice,
                and made the common abstraction so much easier.

            3. Different generators representing different launches.
                This helps keep the state management clean.

            4. This means that the ``Generator`` returned can be customized / contexualized.
                Since ``__next__`` is top level, they cannot be customized in a local manner,
                because decorators would take parameters from the global scope,

            5. This means that we would not have to use double recursion to provide local context,
                because ``__iter__`` can do it and the configs doesn't pollute the global.
                Contexts of these are essential for memoization and remote execution.

            The main downside, I would say, as I have run into this,
            is that since ``Generator``s are lazy, writing eager computation with those is quite difficult.
            For example, ``Generator`` for DAG execution is difficult, as DAG is highly imperative.

            After all, they are pretty much the same,
            so I prefer the ``__iter__`` / ``Generator`` based method as it's cleaner.
        """

    def thunk(self, *ops: Thunk) -> Thunk:
        """
        Convert the current ``Plan`` into a ``Thunk``, wrapping the operands.

        ``Thunk``s preserve the evaluation tree, and can be manipulated during compilation.
        """

        return thunks.thunk(self, *ops)

    @abc.abstractmethod
    def accept[V](self, visitor: Visitor[V]) -> V:
        """
        Must call either ``op_0``, ``op_1``, ``op_2``.
        """

        ...


class Plan0(Plan, ABC):
    ARGC = 0

    @typing.final
    @typing.override
    def apply(self) -> BatchGen:
        yield from self.stream()

    @abc.abstractmethod
    def stream(self) -> BatchGen:
        """
        Yield the stream of ``Block``s.
        """

        ...

    @typing.override
    def accept[T](self, visitor: Plan.Visitor[T]) -> T:
        return visitor.plan_0(self)


class Plan1(Plan, ABC):
    ARGC = 1

    @abc.abstractmethod
    def apply(self, stream_iter: BatchIter, /) -> BatchGen: ...

    @typing.override
    def accept[T](self, visitor: Plan.Visitor[T]) -> T:
        return visitor.plan_1(self)


class Plan2(Plan, ABC):
    ARGC = 2

    @typing.final
    @typing.override
    def apply(self, left_iter: BatchIter, right_iter: BatchIter, /) -> BatchGen:
        for left, right in self.zip(left_iter, right_iter):
            yield self.join(left, right)

    @abc.abstractmethod
    def zip(
        self, left: BatchIter, right: BatchIter, /
    ) -> Iterator[tuple[TensorDict, TensorDict]]:
        """
        Zip over left and right. Defines different iteration strategy here.
        """

        ...

    @abc.abstractmethod
    def join(self, left: TensorDict, right: TensorDict, /) -> TensorDict:
        """
        Join the blocks that are zipped.
        """

        ...

    @typing.override
    def accept[T](self, visitor: Plan.Visitor[T]) -> T:
        return visitor.plan_2(self)
