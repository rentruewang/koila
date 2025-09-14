# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from abc import ABC

from aioway.blocks import Block

from .ops import BlockGen, BlockIter, Op1

__all__ = ["FlatMapOpBase", "RepeatOp"]


@dcls.dataclass(frozen=True)
class FlatMapOpBase(Op1, ABC):
    @typing.final
    @typing.override
    def apply(self, stream_iter: BlockIter, /) -> BlockGen:
        for block in stream_iter:
            yield from self.flat_map(block)

    @abc.abstractmethod
    def flat_map(self, item: Block, /) -> BlockGen:
        """
        Map individual ``Block`` into something else.
        """

        ...


@dcls.dataclass(frozen=True)
class RepeatOp(FlatMapOpBase, key="REPEAT"):
    """
    ```RepeatOp`` repeats every input ``times`` times.
    """

    times: int = 1
    """
    The number of times to repeat the input.
    """

    @typing.override
    def flat_map(self, item: Block):
        for _ in range(self.times):
            yield item
