# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing

from aioway.blocks import Block

from .ops import Op2


@dcls.dataclass(frozen=True)
class ZipOp(Op2, key="ZIP"):
    """
    The ``ZIP`` operation, similar to how you use the builtin ``zip``.
    """

    @typing.override
    def join(self, left: Block, right: Block) -> Block:
        return left.zip(right)
