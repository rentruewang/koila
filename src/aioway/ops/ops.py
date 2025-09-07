# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import inspect
import logging
import typing
from abc import ABC
from collections.abc import Generator, Iterable, Iterator
from typing import ClassVar, Protocol

from aioway import registries
from aioway.blocks import Block
from aioway.errors import AiowayError

__all__ = ["Op", "Op0", "Op1", "Op2", "BlockIter", "BlockGen"]

LOGGER = logging.getLogger(__name__)

type BlockIter = Iterable[Block]
"""
The function ``__iter__`` of ``Block``.
Would create a new ``Generator`` everytime ``iter`` is called.
"""

type BlockGen = Generator[Block]
"""
``Generator`` of ``Block``.
"""


@dcls.dataclass(frozen=True)
class Op(ABC):
    """
    ``Op`` is the operation, be it operator or operand, that works on data.

    An operation is essentially an iterable over data (``Block``),
    but itself doesn't store which previous node it would be operating on.
    This design allows users to write ``Op`` as generators,
    rather than iterators (``__next__`` function) with state management.

    There are 3 kinds of operators, ``Op0``, ``Op1``, ``Op2``,
    representing 0-ary, 1-ary, 2-ary operators respectively.
    """

    ARGC: ClassVar[int]
    """
    Argument count of the current node.
    """

    class Visitor[T](Protocol):
        """
        The ``Visitor`` pattern / strategy for ``Op``.
        """

        @abc.abstractmethod
        def op_0(self, op: "Op0") -> T: ...

        @abc.abstractmethod
        def op_1(self, op: "Op1") -> T: ...

        @abc.abstractmethod
        def op_2(self, op: "Op2") -> T: ...

    def __init_subclass__(cls, key: str = "") -> None:
        # Allow abstract classes, which would not be initialized,
        # to not define keys, as factories are used to store leaf nodes.
        if inspect.isabstract(cls):
            return

        LOGGER.debug("Initializing %s, whose key=%s", cls, key)

        # Impossible if `nargs_init_subclass` is only called in ``__init_subclass``.
        if not issubclass(cls, Op):
            raise OpSubclassError(
                "`nargs_init_subclass` must be called in `__init_subclass__`."
            )

        # Add to registry.
        registries.init_subclass(lambda: Op)(cls, key=key)

    @typing.no_type_check
    @abc.abstractmethod
    def apply(self, *ops: BlockIter) -> BlockGen:
        """
        The ``apply`` method launches a new ``Generator`` to loop over the inputs.
        Every call is creates / rebuilds brand new computation.

        Using ``Generator`` to allow possible future 2 way communication into coroutine.

        We want the subclasses signatures to be::

            #. For ``op0: Op0``, ``op0.apply()``.
            #. For ``op1: Op1``, ``op1(child)``.
            #. For ``op2: Op2``, ``op2(left, right)``.

        Args:
            *ops: The operands to take (would be unpacked in subclass).

        Returns:
            A stream of ``Block``s.
        """

        ...

    def thunk(self, *ops: BlockIter):
        """
        Convert the current ``Op`` into an ``Iterable``, wrapping the operands.

        This is a shortcut function to create an ``Exec``.
        """

        from aioway.execs import Exec

        return Exec(self, *ops)

    @abc.abstractmethod
    def accept[V](self, visitor: Visitor[V]) -> V:
        """
        Must call either ``op_0``, ``op_1``, ``op_2``.
        """

        ...


class Op0(Op, ABC):
    @typing.final
    @typing.override
    def apply(self) -> BlockGen:
        yield from self.stream()

    @abc.abstractmethod
    def stream(self) -> BlockGen:
        """
        Yield the stream of ``Block``s.
        """

        ...

    @typing.override
    def accept[T](self, visitor: Op.Visitor[T]) -> T:
        return visitor.op_0(self)


class Op1(Op, ABC):
    @typing.final
    @typing.override
    def apply(self, stream_iter: BlockIter, /) -> BlockGen:
        for block in stream_iter:
            yield self.map(block)

    @abc.abstractmethod
    def map(self, item: Block, /) -> Block:
        """
        Map individual ``Block`` into something else.
        """

        ...

    @typing.override
    def accept[T](self, visitor: Op.Visitor[T]) -> T:
        return visitor.op_1(self)


class Op2(Op, ABC):
    @typing.final
    @typing.override
    def apply(self, left_iter: BlockIter, right_iter: BlockIter, /) -> BlockGen:
        for left, right in self.zip(left_iter, right_iter):
            yield self.join(left, right)

    @abc.abstractmethod
    def zip(
        self, left: BlockIter, right: BlockIter, /
    ) -> Iterator[tuple[Block, Block]]:
        """
        Zip over left and right. Defines different iteration strategy here.
        """

        ...

    @abc.abstractmethod
    def join(self, left: Block, right: Block, /) -> Block:
        """
        Join the blocks that are zipped.
        """

        ...

    @typing.override
    def accept[T](self, visitor: Op.Visitor[T]) -> T:
        return visitor.op_2(self)


class OpInitError(AiowayError, TypeError): ...


class OpSubclassError(AiowayError, RuntimeError): ...
