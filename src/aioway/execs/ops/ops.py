# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from abc import ABC

from aioway import registries
from aioway.blocks import Block
from aioway.execs.batches import Batch, Batch0, Batch1, Batch2

__all__ = ["Op", "Op0", "Op1", "Op2"]


@dcls.dataclass(frozen=True)
class Op[B: Batch](ABC):
    """
    ``Op`` is the (pure) operator that is responsible for processing
    the input ``Block``s (in the form of ``Batch``) and convert them into an output ``Block``.

    Potentially, ``Op`` would need to be sent over the internet.
    """

    def __init_subclass__(cls, *, key: str = ""):
        registries.init_subclass(lambda: Op)(cls, key=key)

    @abc.abstractmethod
    def __call__(self, batch: B, /) -> Block: ...


@dcls.dataclass(frozen=True)
class Op0(Op[Batch0], ABC):
    """
    ``Op0`` is an 0-ary ``Op``. Essentially an ``Iterator[Block]``.
    """

    @typing.override
    def __call__(self, batch: Batch0) -> Block:
        _ = batch
        return next(self)

    @abc.abstractmethod
    def __next__(self) -> Block: ...


@dcls.dataclass(frozen=True)
class Op1(Op[Batch1], ABC):
    """
    ``Op1`` is an 1-ary ``Op``.
    """

    @typing.override
    def __call__(self, batch: Batch1) -> Block:
        [child] = batch
        return self.map(child)

    @abc.abstractmethod
    def map(self, child: Block, /) -> Block: ...


@dcls.dataclass(frozen=True)
class Op2(Op[Batch2], ABC):
    """
    ``Op2`` is an 2-ary ``Op``.
    """

    @typing.override
    def __call__(self, batch: Batch2) -> Block:
        [left, right] = batch
        return self.join(left, right)

    @abc.abstractmethod
    def join(self, left: Block, right: Block, /) -> Block: ...
