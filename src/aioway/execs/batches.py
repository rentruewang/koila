# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from abc import ABC
from collections.abc import Iterator

from aioway.blocks import Block
from aioway.errors import AiowayError
from aioway.execs.utils import Nargs

__all__ = ["Batch", "Batch0", "Batch1", "Batch2"]


@dcls.dataclass(frozen=True)
class Batch(Nargs, ABC):
    """
    A batch is simply a collection of blocks.
    It is used to represent the variety of output data extracted from ``Poller``.
    """

    def __init_subclass__(cls, key: str):
        cls._init_subclass(Batch, key=key)

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Block]: ...

    def accept[T](self, visitor: "BatchVisitor[T]", /) -> T:
        if not isinstance(visitor, BatchVisitor):
            raise NotBatchVisitorError(f"Not a batch visitor. Got {type(visitor)=}")

        return self._accept(visitor)

    @abc.abstractmethod
    def _accept[T](self, visitor: "BatchVisitor[T]", /) -> T: ...


class BatchVisitor[T](ABC):
    """
    The visitor class for ``Batch``.
    """

    @abc.abstractmethod
    def nullary(self, batch: "Batch0", /) -> T: ...

    @abc.abstractmethod
    def unary(self, batch: "Batch1", /) -> T: ...

    @abc.abstractmethod
    def binary(self, batch: "Batch2", /) -> T: ...


@dcls.dataclass(frozen=True)
class Batch0(Batch, key="_0"):
    ARGC = 0

    @typing.override
    def __iter__(self):
        return
        yield

    @typing.override
    def _accept[T](self, visitor: BatchVisitor[T]) -> T:
        return visitor.nullary(self)


@dcls.dataclass(frozen=True)
class Batch1(Batch, key="_1"):
    ARGC = 1

    block: Block

    @typing.override
    def __iter__(self):
        yield self.block

    @typing.override
    def _accept[T](self, visitor: BatchVisitor[T]) -> T:
        return visitor.unary(self)


@dcls.dataclass(frozen=True)
class Batch2(Batch, key="_2"):
    ARGC = 2

    left: Block
    right: Block

    @typing.override
    def __iter__(self):
        yield self.left
        yield self.right

    @typing.override
    def _accept[T](self, visitor: BatchVisitor[T]) -> T:
        return visitor.binary(self)


class NotBatchVisitorError(AiowayError, TypeError): ...
