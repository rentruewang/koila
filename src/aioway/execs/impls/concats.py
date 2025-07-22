# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import functools
import typing
from collections.abc import Iterator

from aioway.blocks import Block
from aioway.errors import AiowayError
from aioway.execs.execs import Exec

from .binary import BinaryExec

__all__ = ["ZipExec"]


@typing.final
@dcls.dataclass
class ZipExec(BinaryExec, key="ZIP"):
    """
    ``ZipExec`` merges 2 ``Exec``s that have identical length together.
    """

    @typing.override
    def __next__(self) -> Block:
        left, right = next(self._iterator)
        return left.zip(right)

    @functools.cached_property
    def _iterator(self) -> Iterator[tuple[Block, Block]]:
        return zip_over(self.left, self.right)


def zip_over(left: Exec, right: Exec):
    for l, r in zip(left, right):
        yield l, r


class ConcatLengthMismatchError(AiowayError, TypeError): ...
