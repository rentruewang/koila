# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import functools
import itertools
import typing

from aioway.blocks import Block

from .unary import UnaryExec


@dcls.dataclass
class RepeatExec(UnaryExec, key="REPEAT"):
    times: int

    @typing.override
    def __next__(self) -> Block:
        return next(self._generator)

    @functools.cached_property
    def _generator(self):
        for block, _ in itertools.product(self.child, range(self.times)):
            yield block
