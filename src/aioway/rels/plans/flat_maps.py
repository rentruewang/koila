# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from abc import ABC

from tensordict import TensorDict

from .ops import BatchGen, BatchIter, Plan1

__all__ = ["FlatMapPlanBase", "RepeatPlan"]


@dcls.dataclass(frozen=True)
class FlatMapPlanBase(Plan1, ABC):
    @typing.final
    @typing.override
    def apply(self, stream_iter: BatchIter, /) -> BatchGen:
        for block in stream_iter:
            yield from self.flat_map(block)

    @abc.abstractmethod
    def flat_map(self, item: TensorDict, /) -> BatchGen:
        """
        Map individual ``Block`` into something else.
        """

        ...


@dcls.dataclass(frozen=True)
class RepeatPlan(FlatMapPlanBase):
    """
    ```RepeatPlan`` repeats every input ``times`` times.
    """

    times: int = 1
    """
    The number of times to repeat the input.
    """

    @typing.override
    def flat_map(self, item: TensorDict):
        for _ in range(self.times):
            yield item
