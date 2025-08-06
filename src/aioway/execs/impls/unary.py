# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import functools
from abc import ABC
from collections.abc import Iterator

from aioway.blocks import Block
from aioway.execs.execs import Execution
from aioway.nodes import UnaryNode

__all__ = ["UnaryExec"]


@dcls.dataclass
class UnaryExec(Execution, UnaryNode, ABC):
    """
    ``UnaryExec`` is a base class for all unary operations.
    """

    child: Execution
    """
    The input ``Exec`` of the current ``Exec``.
    """

    @staticmethod
    def pass_through(exe: Execution):
        yield from exe

    @functools.cached_property
    def _simple_iterator(self) -> Iterator[Block]:
        return UnaryExec.pass_through(self.child)
