# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import functools
import typing
from collections.abc import Iterator

from aioway.attrs import AttrSet
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

    def __post_init__(self) -> None:
        # Check intersection with the logic in `TableSchema.__and__`.
        _ = self.left.attrs & self.right.attrs

    @typing.override
    def __next__(self) -> Block:
        left, right = next(self._iterator)
        return left.zip(right)

    @functools.cached_property
    def _iterator(self) -> Iterator[tuple[Block, Block]]:
        return zip_over(self.left, self.right)

    @property
    @typing.override
    def attrs(self) -> AttrSet:
        return self.left.attrs | self.right.attrs


def zip_over(left: Exec, right: Exec):
    for l, r in zip(left, right):
        yield l, r


class ConcatLengthMismatchError(AiowayError, TypeError): ...
