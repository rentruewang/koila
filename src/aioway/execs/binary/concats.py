# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing

from aioway.attrs import AttrSet
from aioway.blocks import Block
from aioway.errors import AiowayError

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
        left = next(self.left)
        right = next(self.right)
        return left.zip(right)

    @property
    @typing.override
    def attrs(self) -> AttrSet:
        return self.left.attrs | self.right.attrs


class ConcatLengthMismatchError(AiowayError, TypeError): ...
