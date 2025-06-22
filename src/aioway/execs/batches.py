# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from abc import ABC
from collections.abc import Iterator

from aioway.blocks import Block
from aioway.errors import AiowayError

__all__ = ["Batch", "NullaryBatch", "UnaryBatch", "BinaryBatch", "AryBatch"]


@dcls.dataclass(frozen=True)
class Batch(ABC):
    """
    A batch is simply a collection of blocks.
    It is used to represent the variety of output data extracted from ``Poller``.
    """

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
    def nullary(self, batch: "NullaryBatch", /) -> T: ...

    @abc.abstractmethod
    def unary(self, batch: "UnaryBatch", /) -> T: ...

    @abc.abstractmethod
    def binary(self, batch: "BinaryBatch", /) -> T: ...


@dcls.dataclass(frozen=True)
class NullaryBatch(Batch):
    @typing.override
    def __iter__(self):
        return
        yield

    @typing.override
    def _accept[T](self, visitor: BatchVisitor[T]) -> T:
        return visitor.nullary(self)


@dcls.dataclass(frozen=True)
class UnaryBatch(Batch):
    block: Block

    @typing.override
    def __iter__(self):
        yield self.block

    @typing.override
    def _accept[T](self, visitor: BatchVisitor[T]) -> T:
        return visitor.unary(self)


@dcls.dataclass(frozen=True)
class BinaryBatch(Batch):
    left: Block
    right: Block

    @typing.override
    def __iter__(self):
        yield self.left
        yield self.right

    @typing.override
    def _accept[T](self, visitor: BatchVisitor[T]) -> T:
        return visitor.binary(self)


type AryBatch = NullaryBatch | UnaryBatch | BinaryBatch


class NotBatchVisitorError(AiowayError, TypeError): ...
